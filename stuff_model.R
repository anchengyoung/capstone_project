library(tidyverse)

data_build <- read.csv("statcast_20_21.csv")
data_apply <- read.csv("statcast_2022.csv")

data <- data_build %>%
  mutate(pitch_id = row_number()) %>%
  select(pitch_id, pitch_type, game_date, release_speed, release_pos_x, release_pos_z,
         batter, pitcher, type, balls, strikes, pfx_x, pfx_z, release_spin_rate,
         release_extension, estimated_woba_using_speedangle, woba_value, game_pk) %>%
  rename(bb_type = type)

write.csv(data, file = "data.csv", row.names = FALSE)

## ------------------------------------------------------------------------------------------------

data_build <- data_build %>%
  select(-spin_dir, -umpire, -sv_id, -ends_with("deprecated")) %>%
  # Give each pitch a unique ID
  mutate(pitch_id = row_number(),
         # Create pitch type groupings
         pitch_type_condensed = case_when(pitch_type %in% c("KC","CS") ~ "CU",
                                          pitch_type == "FS" ~ "CH",
                                          pitch_type == "FA" ~ "FF",
                                          pitch_type %in% c("KN","EP","SC","") ~ "NA",
                                          TRUE ~ pitch_type),
         # Combine ball and strike count
         count = paste(balls, strikes, sep = "-"),
         end_count = case_when(type == "X" ~ count,
                               type == "B" ~ paste((balls+1), strikes, sep = "-"),
                               type == "S" & strikes == 2 & description == "foul" ~ count,
                               TRUE ~ paste(balls, (strikes+1), sep = "-")),
         pfx_x = ifelse(p_throws == "L", pfx_x * -1, pfx_x),
         release_pos_x = ifelse(p_throws == "L", release_pos_x * -1, release_pos_x),
         plate_x = ifelse(p_throws == "L", plate_x * -1, plate_x),
         spin_axis = ifelse(p_throws == "L", 360 - spin_axis, spin_axis)) %>%
  filter(pitch_type_condensed != "NA") %>%
  select(pitch_id, everything())

# Do the same with 2022 data
data_apply <- data_apply %>%
  select(-spin_dir, -umpire, -sv_id, -ends_with("deprecated")) %>%
  # Give each pitch a unique ID
  mutate(pitch_id = row_number(),
         # Create pitch type groupings
         pitch_type_condensed = case_when(pitch_type %in% c("KC","CS") ~ "CU",
                                          pitch_type == "FS" ~ "CH",
                                          pitch_type == "FA" ~ "FF",
                                          pitch_type %in% c("KN","EP","SC","") ~ "NA",
                                          TRUE ~ pitch_type),
         count = paste(balls, strikes, sep = "-"),
         end_count = case_when(type == "X" ~ count,
                               type == "B" ~ paste((balls+1), strikes, sep = "-"),
                               type == "S" & strikes == 2 & description == "foul" ~ count,
                               TRUE ~ paste(balls, (strikes+1), sep = "-")),
         pfx_x = ifelse(p_throws == "L", pfx_x * -1, pfx_x),
         release_pos_x = ifelse(p_throws == "L", release_pos_x * -1, release_pos_x),
         plate_x = ifelse(p_throws == "L", plate_x * -1, plate_x),
         spin_axis = ifelse(p_throws == "L", 360 - spin_axis, spin_axis)) %>%
  filter(pitch_type_condensed != "NA") %>%
  select(pitch_id, everything())

# 80/20 train-test split
smp_size <- floor(0.8 * nrow(data_build))
set.seed(42)
train_ind <- sample(seq_len(nrow(data_build)), size = smp_size)
train <- data_build[train_ind, ]
test <- data_build[-train_ind, ]

## ------------------------------------------------------------------------------------------------

# Select only BBE from training data
bbe_df <- train %>% filter(!is.na(estimated_woba_using_speedangle))
unique(bbe_df$count)

# Remove pitches with incorrect count due to umpire mistake
bbe_df <- bbe_df %>% filter(balls < 4 & strikes < 3)
unique(bbe_df$count)

count_xwOBA <- bbe_df %>% group_by(count) %>%
  summarize(n = n(), xwOBA = mean(estimated_woba_using_speedangle, na.rm = TRUE))

## ------------------------------------------------------------------------------------------------

count_xwOBA <- count_xwOBA %>% select(-n)

# Create target y variable: delta_xwOBA
train <- train %>%
  # Create start_xwOBA
  left_join(count_xwOBA, by = "count") %>%
  rename(start_xwOBA = xwOBA) %>%
  # Create end_xwOBA
  left_join(count_xwOBA %>%
              rename(end_count = count), by = "end_count") %>%
  rename(end_xwOBA = xwOBA) %>%
  mutate(end_xwOBA = case_when(type == "X" ~ estimated_woba_using_speedangle,
                               events %in% c("walk","hit_by_pitch","catcher_interf",
                                             "strikeout","strikeout_double_play") ~ woba_value,
                               TRUE ~ end_xwOBA),
         delta_xwOBA = end_xwOBA - start_xwOBA) %>% filter(!is.na(delta_xwOBA))

## ------------------------------------------------------------------------------------------------

# Do the same with the testing set
test <- test %>%
  # Create start_xwOBA
  left_join(count_xwOBA, by = "count") %>%
  rename(start_xwOBA = xwOBA) %>%
  # Create end_xwOBA
  left_join(count_xwOBA %>%
              rename(end_count = count), by = "end_count") %>%
  rename(end_xwOBA = xwOBA) %>%
  mutate(end_xwOBA = case_when(type == "X" ~ estimated_woba_using_speedangle,
                               events %in% c("walk","hit_by_pitch","catcher_interf",
                                             "strikeout","strikeout_double_play") ~ woba_value,
                               TRUE ~ end_xwOBA),
         delta_xwOBA = end_xwOBA - start_xwOBA) %>% filter(!is.na(delta_xwOBA))

# 50/50 test-val split
# smp_size_2 <- floor(0.5 * nrow(test))
# set.seed(42)
# test_ind <- sample(seq_len(nrow(test)), size = smp_size_2)
# val <- test[-test_ind, ]
# test <- test[test_ind, ]

## ------------------------------------------------------------------------------------------------

library(xgboost)
# Features that go in the model
features <- c("pitch_type_condensed","release_speed",
              "pfx_x","pfx_z","release_extension")

# Prepare the Training set to train the model
train_feature <- train %>% select(pitch_id, all_of(features), delta_xwOBA)
anyNA(train_feature)
train_feature <- drop_na(train_feature)

# Prepare the Testing set to test the model
test_feature <- test %>% select(pitch_id, all_of(features), delta_xwOBA)
anyNA(test_feature)
test_feature <- drop_na(test_feature)

## ------------------------------------------------------------------------------------------------

train_errors <- data.frame(pitch_type = character(),
                           RMSE = double(),
                           R_squared = double())

test_errors <- data.frame(pitch_type = character(),
                          RMSE = double(),
                          R_squared = double())

for (pitch_type in unique(train_feature$pitch_type_condensed)) {
  
  train_pitch <- train_feature %>%
    filter(pitch_type_condensed == pitch_type) %>%
    select(-pitch_type_condensed)
  
  train_data <- as.matrix(train_pitch %>% select(-pitch_id, -delta_xwOBA))
  train_label <- as.matrix(train_pitch %>% select(delta_xwOBA))
  xgb_train <- xgb.DMatrix(data = train_data, label = train_label)
  
  #xgb_model <- xgb.cv(data = xgb_train, max.depth = 6, nrounds = 100, nfold = 5,
  #                    early_stopping_rounds = 3, objective = "reg:squarederror")
  
  xgb_model_pitch <- xgboost(data = xgb_train, max.depth = 6,
                             nrounds = 5, objective = "reg:squarederror")
  
  assign(paste0("xgb_model_", tolower(pitch_type)), xgb_model_pitch, envir = .GlobalEnv)
  
  train_pitch <- cbind(train_pitch, data.frame(pred_xwOBA = predict(xgb_model_pitch,
                                                                    newdata = train_data)))
  
  assign(paste0("train_", tolower(pitch_type)), train_pitch, envir = .GlobalEnv)
  
  train_rmse <- round(Metrics::rmse(train_pitch$delta_xwOBA, train_pitch$pred_xwOBA), 3)
  train_r_squared <- round(cor(train_pitch$delta_xwOBA, train_pitch$pred_xwOBA)^2, 5)
  
  train_errors <- data.table::rbindlist(list(train_errors,
                                             as.list(c(pitch_type, train_rmse, train_r_squared))))
  
  test_pitch <- test_feature %>%
    filter(pitch_type_condensed == pitch_type) %>%
    select(-pitch_type_condensed)
  
  test_data <- as.matrix(test_pitch %>% select(-pitch_id, -delta_xwOBA))
  
  test_pitch <- cbind(test_pitch, data.frame(pred_xwOBA = predict(xgb_model_pitch,
                                                                  newdata = test_data)))
  
  assign(paste0("test_", tolower(pitch_type)), test_pitch, envir = .GlobalEnv)
  
  test_rmse <- round(Metrics::rmse(test_pitch$delta_xwOBA, test_pitch$pred_xwOBA), 3)
  test_r_squared <- round(cor(test_pitch$delta_xwOBA, test_pitch$pred_xwOBA)^2, 5)
  
  test_errors <- data.table::rbindlist(list(test_errors,
                                            as.list(c(pitch_type, test_rmse, test_r_squared))))
  
}

## ------------------------------------------------------------------------------------------------

# Combine
test_all <- rbind(test_ff, test_si, test_fc, test_sl, test_cu, test_ch)
test <- test %>%
  left_join(test_all %>% select(pitch_id, pred_xwOBA), by = "pitch_id") %>%
  group_by(pitch_type_condensed) %>%
  mutate(stuff_plus = (((pred_xwOBA - mean(pred_xwOBA, na.rm = T)) / sd(pred_xwOBA, na.rm = T)) - 1) * -100) %>%
  ungroup() %>% filter(!is.na(pred_xwOBA))

train_all <- rbind(train_ff, train_si, train_fc, train_sl, train_cu, train_ch)
train <- train %>%
  left_join(train_all %>% select(pitch_id, pred_xwOBA), by = "pitch_id") %>%
  group_by(pitch_type_condensed) %>%
  mutate(stuff_plus = (((pred_xwOBA - mean(pred_xwOBA, na.rm = T)) / sd(pred_xwOBA, na.rm = T)) - 1) * -100) %>%
  ungroup() %>% filter(!is.na(pred_xwOBA))

averages <- train %>%
  group_by(pitch_type_condensed) %>%
  summarize(avg_stuff = mean(pred_xwOBA, na.rm = TRUE),
            sd_stuff = sd(pred_xwOBA, na.rm = TRUE))

write.csv(averages, file = "model_stats.csv", row.names = FALSE)

# testing <- test %>%
#   group_by(pitch_type_condensed) %>%
#   mutate(stuff_plus = pred_xwOBA / mean(pred_xwOBA, na.rm = TRUE) * 100) %>%
#   ungroup() %>%
#   filter(pitch_type_condensed == "FF") %>%
#   group_by(pitcher) %>%
#   summarize(name = first(player_name), pitches = n(),
#             velo = mean(release_speed, na.rm = TRUE),
#             spin = mean(release_spin_rate, na.rm = TRUE),
#             VB = mean(pfx_z, na.rm = TRUE),
#             HB = mean(pfx_x, na.rm = TRUE),
#             ext = mean(release_extension, na.rm = TRUE),
#             delta_xwOBA = mean(delta_xwOBA, na.rm = TRUE),
#             pred_xwOBA = mean(pred_xwOBA, na.rm = TRUE),
#             stuff_plus = round(mean(stuff_plus, na.rm = TRUE), 2)) %>%
#   filter(pitches >= 100) %>%
#   arrange(desc(stuff_plus))

# data_apply %>%
#   group_by(pitch_type_condensed) %>%
#   summarize(pitches = n(),
#             delta_xwOBA = mean(delta_xwOBA, na.rm = TRUE),
#             pred_xwOBA = mean(pred_xwOBA, na.rm = TRUE)) %>%
#   arrange(desc(pitches))

## ------------------------------------------------------------------------------------------------

for (pitch_type in unique(apply_df$pitch_type_condensed)) {
  
  df <- apply_df %>%
    filter(pitch_type_condensed == pitch_type)
  
  data <- as.matrix(df %>% select(-pitch_id, -pitch_type_condensed))
  
  xgb_model <- eval(parse(text = paste0("xgb_model_", tolower(pitch_type))))
  
  df <- cbind(df, data.frame(pred_xwOBA = predict(xgb_model, newdata = data)))
  df <- df %>%
    mutate(stuff_plus = (((pred_xwOBA - mean(pred_xwOBA)) / sd(pred_xwOBA)) - 1) * -100)
  
  applied <- rbind(applied, df)
  
}

## ------------------------------------------------------------------------------------------------

# Apply models to 2022 pitches
apply_df <- data_apply %>% select(pitch_id, all_of(features))
anyNA(apply_df)
apply_df <- drop_na(apply_df)

applied <- data.frame()

for (pitch_type in unique(apply_df$pitch_type_condensed)) {
  
  df <- apply_df %>%
    filter(pitch_type_condensed == pitch_type)
  
  data <- as.matrix(df %>% select(-pitch_id, -pitch_type_condensed))
  
  xgb_model <- eval(parse(text = paste0("xgb_model_", tolower(pitch_type))))
  
  df <- cbind(df, data.frame(pred_xwOBA = predict(xgb_model, newdata = data)))
  df <- df %>%
    mutate(stuff_plus = (((pred_xwOBA - mean(pred_xwOBA)) / sd(pred_xwOBA)) - 1) * -100)
  
  applied <- rbind(applied, df)
  
}

#data_apply <- data_apply %>% select(-pred_xwOBA, -stuff_plus)

data_apply <- data_apply %>%
  left_join(applied %>% select(pitch_id, pred_xwOBA, stuff_plus),
            by = "pitch_id")

results <- data_apply %>%
  group_by(pitcher, pitch_type_condensed) %>%
  summarize(name = first(player_name), pitches = n(),
            velo = mean(release_speed, na.rm = TRUE),
            spin = mean(release_spin_rate, na.rm = TRUE),
            VB = mean(pfx_z, na.rm = TRUE),
            HB = mean(pfx_x, na.rm = TRUE),
            ext = mean(release_extension, na.rm = TRUE),
            #delta_xwOBA = mean(delta_xwOBA, na.rm = TRUE),
            #pred_xwOBA = mean(pred_xwOBA, na.rm = TRUE),
            stuff_avg = round(mean(stuff_plus, na.rm = TRUE), 2),
            stuff_sd = round(sd(stuff_plus, na.rm = TRUE), 2)) %>%
  filter(pitches >= 200) %>%
  arrange(desc(stuff_avg)) %>% ungroup() %>%
  group_by(pitch_type_condensed) %>%
  mutate(rank = row_number())

data_apply <- data_apply %>%
  arrange(desc(pitch_id)) %>%
  group_by(pitcher, pitch_type_condensed) %>%
  mutate(pitch_num = row_number())

write.csv(data_apply, "stuff_2022.csv", row.names = FALSE)

## ------------------------------------------------------------------------------------------------

#master <- read.csv("https://raw.githubusercontent.com/spilchen/baseball_id_db/main/master.csv")

steamer <- read.csv("steamer_2022_batting.csv")

steamer <- steamer %>%
  filter(pn == 1, split == "overall") %>%
  mutate(Team = case_when(Team == "CHA" ~ "CHW", Team == "CHN" ~ "CHC",
                          Team == "KCA" ~ "KCR", Team == "LAN" ~ "LAD",
                          Team == "NYA" ~ "NYY", Team == "NYN" ~ "NYM",
                          Team == "SDN" ~ "SDP", Team == "SFN" ~ "SFG",
                          Team == "SLN" ~ "STL", Team == "TBA" ~ "TBR",
                          Team == "WAS" ~ "WSN", TRUE ~ Team),
         firstname = case_when(firstname == "Nicholas" & lastname == "Allen" ~ "Nick",
                               firstname == "Raymond" & lastname == "McDonald" ~ "Mickey",
                               firstname == "Mervyl" & lastname == "Melendez" ~ "MJ",
                               firstname == "Noah" & lastname == "Naylor" ~ "Bo",
                               TRUE ~ firstname),
         steamerid = case_when(lastname == "Capel" & Team == "STL" ~ "19983",
                               lastname == "Call" & Team == "CLE" ~ "19296",
                               lastname == "Waters" & Team == "ATL" ~ "20505",
                               lastname == "O'Hoppe" & Team == "PHI" ~ "24729",
                               lastname == "Groshans" & Team == "TOR" ~ "23794",
                               lastname == "Proctor" & Team == "TBR" ~ "21634",
                               lastname == "Steer" & Team == "MIN" ~ "26323",
                               lastname == "Stevenson" & Team == "TBR" ~ "22411",
                               lastname == "Davis" & Team == "LAA" ~ "18587",
                               lastname == "Bannon" & Team == "BAL" ~ "19768",
                               lastname == "MacKinnon" & Team == "LAA" ~ "19625",
                               lastname == "Papierski" & Team == "HOU" ~ "21386",
                               firstname == "Esteury" & Team == "SDP" ~ "21780",
                               TRUE ~ steamerid),
         wRC_plus = ((wRAA/PA)+(sum(R)/sum(PA)))/(sum(wRC)/sum(PA))*100)

batters <- read.csv("wRC+_2022.csv")

batters <- batters %>%
  filter(PA > 0) %>%
  select(playerid, Name, Team, PA, Bat, wRC.) %>%
  mutate(playerid = as.character(playerid)) %>%
  rename(actual_wRC = wRC.) %>%
  left_join(steamer %>%
              select(steamerid, mlbamid, wRC_plus) %>%
              rename(playerid = steamerid, mlb_id = mlbamid,
                     proj_wRC = wRC_plus), by = "playerid")

batters <- batters %>%
  left_join(batters %>% filter(is.na(proj_wRC)) %>%
              separate(Name, c("firstname","lastname"), " ", extra = "drop") %>%
              left_join(steamer %>%
                          select(firstname, lastname, Team, mlbamid, wRC_plus),
                        by = c("firstname","lastname","Team")) %>%
  select(playerid, mlbamid, wRC_plus), by = "playerid") %>%
  mutate(proj_wRC = ifelse(is.na(proj_wRC), wRC_plus, proj_wRC),
         mlb_id = ifelse(is.na(mlb_id), mlbamid, mlb_id))

batters <- batters %>%
  select(-mlbamid, -wRC_plus) %>%
  left_join(batters %>% filter(is.na(proj_wRC)) %>%
              select(-mlbamid, -wRC_plus) %>%
              separate(Name, c("firstname","lastname"), " ", extra = "drop") %>%
              left_join(steamer %>%
                          select(lastname, Team, mlbamid, wRC_plus),
                        by = c("lastname","Team")) %>%
              select(playerid, mlbamid, wRC_plus), by = "playerid") %>%
  mutate(proj_wRC = ifelse(is.na(proj_wRC), wRC_plus, proj_wRC),
         mlb_id = ifelse(is.na(mlb_id), mlbamid, mlb_id)) %>%
  select(-mlbamid, -wRC_plus) %>%
  filter(!is.na(proj_wRC)) %>% arrange(desc(proj_wRC))


# rmse_list <- c()
# 
# for (i in seq(500)) {
#   
#   df <- batters %>% filter(PA >= i)
#   rmse <- Metrics::rmse(df$actual_wRC, df$proj_wRC)
#   rmse_list <- c(rmse_list, rmse)
#   
# }
# 
# rmse_df <- data.frame(rmse = rmse_list) %>%
#   mutate(PA = row_number())
# 
# ggplot(rmse_df, aes(x = PA, y = rmse)) + geom_line()

#ggplot(batters, aes(x = proj_wRC, fill = cluster)) + geom_histogram(binwidth = 1)

# mean(batters$actual_wRC)
# median(batters$actual_wRC)
# mean(batters$proj_wRC)
# median(batters$proj_wRC)
# 
# batters %>%
#   mutate(total_wRC = PA * actual_wRC) %>%
#   summarize(total_wRC = sum(total_wRC),
#             total_PA = sum(PA),
#             average = total_wRC / total_PA)

## ------------------------------------------------------------------------------------------------

#batters <- batters %>% select(-cluster)

to_class <- batters %>% select(proj_wRC, Bat)
set.seed(42)
classifier <- stats::kmeans(to_class, centers = 5, nstart = 50)
batters <- cbind(batters, classifier$cluster) %>%
  rename(cluster = `classifier$cluster`) %>%
  mutate(cluster = case_when(cluster == 1 ~ "Star",
                             cluster == 2 ~ "Role",
                             cluster == 3 ~ "Bench",
                             cluster == 4 ~ "Everyday",
                             cluster == 5 ~ "Replacement"),
         cluster = factor(cluster, levels = c("Replacement","Bench","Role","Everyday","Star")))

batters %>%
  group_by(cluster) %>%
  summarize(n())

ggplot(batters, aes(x = proj_wRC, y = Bat, color = cluster)) +
  geom_point() +
  labs(title = "2022 MLB Batter Clusters by k-means Clustering",
       x = "Pre-Season Projected wRC+", y = "2022 Batting Runs",
       color = "Cluster") + theme(legend.position = "top") +
  scale_color_manual(values = c("red2", "yellow3","green3","dodgerblue","purple3"))

ggplot(batters, aes(x = proj_wRC, fill = cluster)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Batter Clusters by Projected wRC+",
       x = "Pre-Season Projected wRC+", y = "Density", fill = "Cluster") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("red2", "yellow3","green3","dodgerblue","purple3"))


# ggplot(data_apply %>% filter(pitch_type_condensed == "FF"),
#        aes(x = stuff_plus, y = proj_wRC)) + geom_point() + xlim(0, 200)

## ------------------------------------------------------------------------------------------------

data_apply <- data_apply %>%
  left_join(batters %>%
              select(mlb_id, proj_wRC, cluster) %>%
              rename(batter = mlb_id), by = "batter")

summary(data_apply$stuff_plus)

pitchers <- data_apply %>% group_by(pitcher) %>%
  summarize(name = first(player_name),
            stuff_avg = mean(stuff_plus, na.rm = TRUE)) %>%
  arrange(desc(stuff_avg))

data_apply <- data_apply %>%
  left_join(pitchers, by = "pitcher") %>%
  mutate(adj_stuff = stuff_plus - stuff_avg + 100,
         adj_stuff = ifelse(is.infinite(adj_stuff), NA, adj_stuff))

data_apply %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  summarize(pitches = n(),
            stuff = round(mean(stuff_plus, na.rm = TRUE),2),
            adj_stuff = round(mean(adj_stuff, na.rm = TRUE), 3))

by_PA <- data_apply %>%
  filter(!is.na(cluster)) %>%
  group_by(game_pk, at_bat_number) %>%
  summarize(stuff_avg = mean(stuff_avg, na.rm = TRUE),
            cluster = first(cluster)) %>% ungroup() %>%
  group_by(cluster) %>%
  summarize(pitcher_skill = mean(stuff_avg))

by_PA

facing <- data_apply %>%
  filter(!is.na(cluster)) %>%
  group_by(pitch_type_condensed, cluster) %>%
  summarize(pitch_type_condensed = factor(pitch_type_condensed,
                                          levels = c("FF","SI","FC",
                                                     "SL","CU","CH")),
            stuff_plus = mean(stuff_plus, na.rm = TRUE),
            adj_stuff = mean(adj_stuff, na.rm = TRUE))

ggplot(facing, aes(x = pitch_type_condensed, y = adj_stuff, fill = cluster)) +
  geom_bar(position="dodge", stat="identity") +
  labs(title = "Average Pitcher-Adjusted Stuff+ faced by Batter Clusters",
       subtitle = "on a per-Plate Appearance Basis",
       x = "Pitch Type", y = "Adjusted Stuff+", fill = "Cluster") + 
  theme(legend.position = "bottom")

check <- data_apply %>%
  #filter(cluster == "Star") %>%
  filter(!is.na(cluster)) %>%
  group_by(pitcher, cluster) %>%
  summarize(name = first(player_name),
            pitches = n(),
            stuff = mean(stuff_plus, na.rm = TRUE),
            adj_stuff = mean(adj_stuff, na.rm = TRUE)) %>%
  arrange(desc(pitches))

ggplot(check, aes(x = stuff, y = adj_stuff)) + geom_point() +
  facet_wrap(~ cluster) +  geom_smooth(method = 'lm') +
  xlim(-50, 300) + ylim(-50, 300)

check %>%
  group_by(cluster) %>%
  summarize(R_squared = cor(stuff, adj_stuff)^2)



## ------------------------------------------------------------------------------------------------

## ------------------------------------------------------------------------------------------------

# val_feature <- val %>% select(all_of(features), delta_xwOBA)
# anyNA(val_feature)
# val_feature <- drop_na(val_feature)
# 
# ggplot(val_feature %>% filter(pitch_type_condensed == "SI", delta_xwOBA > 0.1),
#        aes(x = pfx_x, y = pfx_z, color = delta_xwOBA)) +
#   geom_point(alpha = 0.5)
# 
# library(caret)
# train_ctl <- trainControl(method="repeatedcv", number=10, repeats=3)
# set.seed(42)
# model <- caret::train(delta_xwOBA ~ ., data = val_feature, method = "knn",
#                       trControl = train_ctl, tuneLength = 1)
# model
# 
# test <- test %>% drop_na(all_of(features))
# 
# test <- cbind(test, data.frame(pred_xwOBA = predict(model, newdata = test)))
# 
# caret::varImp(model, scale=FALSE) %>% arrange(desc(Overall))

## ------------------------------------------------------------------------------------------------

# ggplot(test, aes(x = delta_xwOBA, y = pred_xwOBA, color = pitch_type_condensed)) +
#   geom_point() + facet_wrap(~pitch_type_condensed) + geom_smooth(method = 'lm')
# 
# ggplot(train, aes(x = delta_xwOBA)) + geom_histogram(bins = 100)

## ------------------------------------------------------------------------------------------------


## ------------------------------------------------------------------------------------------------

# train_feature <- train %>%
#   select(pitch_id, all_of(features), delta_xwOBA)
# anyNA(train_feature)
# train_feature <- drop_na(train_feature)
# 
# train_ff <- train_feature %>%
#   filter(pitch_type_condensed == "FF") %>%
#   select(-pitch_type_condensed)

# library(ranger)
# rf <- ranger(delta_xwOBA ~ ., data = train_ff, num.trees = 1000, oob.error = FALSE)
# 
# test_feature <- test %>% select(all_of(features), delta_xwOBA)
# anyNA(test_feature)
# test_feature <- drop_na(test_feature)
# 
# test_ff <- test_feature %>%
#   filter(pitch_type_condensed == "FF") %>%
#   select(-pitch_type_condensed)
# 
# test_ff <- cbind(test_ff, data.frame(pred_xwOBA = predict(rf, data = test_ff)))
# 
# ggplot(test_ff, aes(x = delta_xwOBA, y = prediction)) +
#   geom_point() + xlim(-0.5, 1.7) + ylim(-0.5, 1.7)
# 
# Metrics::rmse(test_ff$delta_xwOBA, test_ff$prediction)
# cor(test_ff$delta_xwOBA, test_ff$prediction)^2
# 
# 
# train_data <- train_ff %>% select(-delta_xwOBA)
# train_label <- train_ff %>% select(delta_xwOBA)
# 
# library(xgboost)
# xgb_train <- xgb.DMatrix(data = as.matrix(train_data), label = as.matrix(train_label))
# 
# xgb_model <- xgboost(data = xgb_train, max.depth = 6, nrounds = 2,
#                      objective = "reg:squarederror")
# 
# test_data <- test_ff %>% select(-delta_xwOBA)
# #test_label <- test_ff %>% select(delta_xwOBA)
# test_data <- as.matrix(test_data)
# 
# test_ff <- cbind(test_ff, data.frame(pred_xwOBA = predict(xgb_model, newdata = test_data)))
# 
# Metrics::rmse(test_ff$delta_xwOBA, test_ff$pred_xwOBA)
# cor(test_ff$delta_xwOBA, test_ff$pred_xwOBA)^2
# 
# ggplot(test_ff, aes(x = delta_xwOBA, y = pred_xwOBA)) + geom_point()
# 
# ggplot(test_ff, aes(x = delta_xwOBA)) + geom_histogram(bins = 200)
# 
# testing <- test %>%
#   filter(pitch_type_condensed == "FF") %>%
#   left_join(test_ff, by = c("release_speed","pfx_x","pfx_z",
#                             "release_extension","release_spin_rate","delta_xwOBA"))
# 
# results <- testing %>%
#   group_by(pitcher) %>%
#   summarize(name = first(player_name), pitches = n(),
#             velo = mean(release_speed, na.rm = TRUE),
#             spin = mean(release_spin_rate, na.rm = TRUE),
#             VB = mean(pfx_z, na.rm = TRUE),
#             HB = mean(pfx_x, na.rm = TRUE),
#             ext = mean(release_extension, na.rm = TRUE),
#             delta_xwOBA = mean(delta_xwOBA, na.rm = TRUE),
#             #rf_pred = mean(prediction, na.rm = TRUE),
#             pred_xwOBA = mean(pred_xwOBA, na.rm = TRUE)) %>%
#   filter(pitches >= 100) %>%
#   arrange(pred_xwOBA)

## ------------------------------------------------------------------------------------------------

# library(xgboost)
# 
# train_feature <- train %>% select(all_of(features), delta_xwOBA)
# anyNA(train_feature)
# train_feature <- drop_na(train_feature)
# 
# train_feature <- train_feature %>%
#   mutate(is_FF = ifelse(pitch_type_condensed == "FF", 1, 0),
#          is_SI = ifelse(pitch_type_condensed == "SI", 1, 0),
#          is_FC = ifelse(pitch_type_condensed == "FC", 1, 0),
#          is_SL = ifelse(pitch_type_condensed == "SL", 1, 0),
#          is_CU = ifelse(pitch_type_condensed == "CU", 1, 0),
#          is_CH = ifelse(pitch_type_condensed == "CH", 1, 0)) %>%
#   select(-pitch_type_condensed)
# 
# train_data <- train_feature %>% select(-delta_xwOBA)
# train_label <- train_feature %>% select(delta_xwOBA)
# 
# xgb_train <- xgb.DMatrix(data = as.matrix(train_data), label = as.matrix(train_label))
# 
# test_feature <- test_feature %>% select(-pred_xwOBA)
# 
# xgb_model <- xgboost(data = xgb_train, max.depth = 6, nrounds = 2,
#                      objective = "reg:squarederror")
# 
# test_feature <- test %>% select(all_of(features), delta_xwOBA)
# anyNA(test_feature)
# test_feature <- drop_na(test_feature)
# 
# test_feature <- test_feature %>%
#   mutate(is_FF = ifelse(pitch_type_condensed == "FF", 1, 0),
#          is_SI = ifelse(pitch_type_condensed == "SI", 1, 0),
#          is_FC = ifelse(pitch_type_condensed == "FC", 1, 0),
#          is_SL = ifelse(pitch_type_condensed == "SL", 1, 0),
#          is_CU = ifelse(pitch_type_condensed == "CU", 1, 0),
#          is_CH = ifelse(pitch_type_condensed == "CH", 1, 0)) %>%
#   select(-pitch_type_condensed)
# 
# test_data <- test_feature %>% select(-delta_xwOBA)
# test_label <- test_feature %>% select(delta_xwOBA)
# 
# test_data <- as.matrix(test_data)
# 
# test_feature <- cbind(test_feature,
#                       data.frame(pred_xwOBA = predict(xgb_model, newdata = test_data)))
# 
# test_feature <- test_feature %>%
#   mutate(pitch_type = case_when(is_FF == 1 ~ "FF",
#                                 is_SI == 1 ~ "SI",
#                                 is_FC == 1 ~ "FC",
#                                 is_SL == 1 ~ "SL",
#                                 is_CU == 1 ~ "CU",
#                                 is_CH == 1 ~ "CH"))
# 
# ggplot(test_feature, aes(x = delta_xwOBA, y = prediction, color = pitch_type_condensed)) +
#   geom_point() + facet_wrap(~pitch_type_condensed) + geom_smooth(method = 'lm')

## ------------------------------------------------------------------------------------------------

# library(randomForest)
# 
# rf <- randomForest(delta_xwOBA ~ ., data = train_feature, ntree = 100,
#                    keep.forest=FALSE, importance=TRUE)
# 
# library(ranger)
# rf <- ranger(delta_xwOBA ~ ., data = train_feature, num.trees = 1000, oob.error = FALSE)
# 
# length(rf$predictions)
# 
# test_feature <- cbind(test_feature, data.frame(pred_xwOBA = predict(rf, data = test_feature)))
# 
# library(Rborist)
# 
# rf <- Rborist(as.matrix(train_data), as.matrix(train_label), nTree = 1000)
 
## ------------------------------------------------------------------------------------------------
 
# model <- lm(delta_xwOBA ~ release_speed + release_spin_rate + spin_axis + release_extension + pfx_x + pfx_z,
#             data = train)
# summary(model)
# 
# val <- cbind(val, predicted = predict(model, newdata = val))
# val <- val %>% filter(!is.na(predicted))
# 
# Metrics::rmse(val$delta_xwOBA, val$predicted)
# cor(val$delta_xwOBA, val$predicted)^2
# 
# model_type <- lm(delta_xwOBA ~ pitch_type + release_speed + release_spin_rate + spin_axis + release_extension + pfx_x + pfx_z,
#                  data = train)
# summary(model_type)
# 
# val <- cbind(val, predicted_2 = predict(model_type, newdata = val))
# val <- val %>% filter(!is.na(predicted_2))
# 
# Metrics::rmse(val$delta_xwOBA, val$predicted_2)
# cor(val$delta_xwOBA, val$predicted_2)^2
# 
# ggplot(val, aes(x = delta_xwOBA, y = predicted_2, color = pitch_type)) +
#   geom_point() + facet_wrap(~pitch_type) + geom_smooth(method = 'lm')

## ------------------------------------------------------------------------------------------------

# colnames(train)
# 
# model <- lm(delta_xwOBA ~ release_speed + release_spin_rate, data = train)
# data_build <- cbind(data_build, predicted = predict(model, newdata = data_build))
# 
# Metrics::rmse(data_build$delta_xwOBA, data_build$predicted)
# 
# ggplot(data_build, aes(x = delta_xwOBA, y = predicted)) +
#   geom_point() + geom_smooth(method = 'lm')

## ------------------------------------------------------------------------------------------------

# data_apply <- data_apply %>%
#   left_join(count_xwOBA, by = "count") %>%
#   rename(start_xwOBA = xwOBA) %>%
#   left_join(count_xwOBA %>%
#               rename(end_count = count), by = "end_count") %>%
#   rename(end_xwOBA = xwOBA) %>%
#   mutate(end_xwOBA = case_when(type == "X" ~ estimated_woba_using_speedangle,
#                                events == "walk" ~ woba_value,
#                                events == "hit_by_pitch" ~ woba_value,
#                                events %in% c("strikeout","strikeout_double_play") ~ woba_value,
#                                TRUE ~ end_xwOBA),
#          delta_xwOBA = end_xwOBA - start_xwOBA) %>%
#   filter(!is.na(release_speed) & !is.na(release_spin_rate) & !is.na(delta_xwOBA))

## ------------------------------------------------------------------------------------------------

# smp_size <- floor(0.5 * nrow(data_apply))
# set.seed(42)
# test_ind <- sample(seq_len(nrow(data_apply)), size = smp_size)
# test <- data_apply[test_ind, ]
# val <- data_apply[-test_ind, ]
# 
# model <- lm(delta_xwOBA ~ release_speed + release_spin_rate, data = data_build)
# val <- val %>% select(-predicted)
# val <- cbind(val, predicted = predict(model, newdata = val))
# 
# Metrics::rmse(val$delta_xwOBA, val$predicted)

## ------------------------------------------------------------------------------------------------

# features <- c("pitch_type_condensed","release_speed",
#               "release_pos_x","release_pos_z",
#               "pfx_x","pfx_z","release_spin_rate",
#               "release_extension","spin_axis")
# 
# val_feature <- val %>% select(features, delta_xwOBA)
# #small_feature <- val_feature %>% select(release_speed, release_spin_rate, delta_xwOBA)
# 
# model <- lm(delta_xwOBA ~ ., data = val_feature)
# caret::varImp(model, scale=FALSE) %>% arrange(desc(Overall))

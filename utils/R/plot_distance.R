library(dplyr)
library(ggplot2)
library(gridExtra)

setwd("/export/storage_covidvaccine")

################################################################

# Total
df <- read.csv("Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/Distance_HPI.csv") %>% filter(Assignment != 0)
df$HPI <- as.factor(df$HPI)
df_rep <- df[rep(seq_len(nrow(df)), df$Assignment), ]

dist_density_blp <- ggplot(df_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = HPI)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot (BLP)") + 
  theme_bw()

ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot.jpg', dist_density, dpi = 300, width = 6, height = 4)

# By pharmacy
df_pharmacy <- df %>% filter(Pharmacy == 1)
df_pharmacy$HPI <- as.factor(df_pharmacy$HPI)
df_pharmacy_rep <- df_pharmacy[rep(seq_len(nrow(df_pharmacy)), df_pharmacy$Assignment), ]

dist_density <- ggplot(df_pharmacy_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = HPI)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot by Pharmacies (BLP)") + 
  theme_bw()

ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot_pharmacy.jpg', dist_density, dpi = 300, width = 6, height = 4)

# By dollar
df_dollar <- df %>% filter(Pharmacy == 0)
df_dollar$HPI <- as.factor(df_dollar$HPI)
df_dollar_rep <- df_dollar[rep(seq_len(nrow(df_dollar)), df_dollar$Assignment), ]

dist_density <- ggplot(df_dollar_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = HPI)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot by Dollar Stores (BLP)") + 
  theme_bw()

ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot_dollar.jpg', dist_density, dpi = 300, width = 6, height = 4)

################################################################

# Total
df <- read.csv("Result/MaxVaxDistLogLin/M5_K8000/Dollar/vaccinated/Distance_HPI.csv") %>% filter(Assignment != 0)
df$HPI <- as.factor(df$HPI)
df_rep <- df[rep(seq_len(nrow(df)), df$Assignment), ]

dist_density_loglinear <- ggplot(df_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = HPI)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot (Log-linear)") + 
  theme_bw()

ggsave('Result/MaxVaxDistLogLin/M5_K8000/Dollar/vaccinated/density_plot.jpg', dist_density, dpi = 300, width = 6, height = 4)

# By pharmacy
df_pharmacy <- df %>% filter(Pharmacy == 1)
df_pharmacy$HPI <- as.factor(df_pharmacy$HPI)
df_pharmacy_rep <- df_pharmacy[rep(seq_len(nrow(df_pharmacy)), df_pharmacy$Assignment), ]

dist_density <- ggplot(df_pharmacy_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = HPI)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot by Pharmacies (Log-linear)") + 
  theme_bw()

ggsave('Result/MaxVaxDistLogLin/M5_K8000/Dollar/vaccinated/density_plot_pharmacy.jpg', dist_density, dpi = 300, width = 6, height = 4)

# By dollar
df_dollar <- df %>% filter(Pharmacy == 0)
df_dollar$HPI <- as.factor(df_dollar$HPI)
df_dollar_rep <- df_dollar[rep(seq_len(nrow(df_dollar)), df_dollar$Assignment), ]

dist_density <- ggplot(df_dollar_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = HPI)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot by Dollar Stores (Log-linear)") + 
  theme_bw()

ggsave('Result/MaxVaxDistLogLin/M5_K8000/Dollar/vaccinated/density_plot_dollar.jpg', dist_density, dpi = 300, width = 6, height = 4)


################################################################

df_BLP <- read.csv("Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/Distance_HPI.csv") %>% filter(Assignment != 0)
df_loglin <- read.csv("Result/MaxVaxDistLogLin/M5_K8000/Dollar/vaccinated/Distance_HPI.csv") %>% filter(Assignment != 0)

df_BLP$Model = 'BLP'
df_loglin$Model = 'LogLinear'
df_agg <- rbind(df_loglin, df_BLP)
df_agg$Model <- as.factor(df_agg$Model)


df_agg$HPI <- as.factor(df_agg$HPI)
df_agg_rep <- df_agg[rep(seq_len(nrow(df_agg)), df_agg$Assignment), ]

dist_density_all <- ggplot(df_agg_rep, aes(x = ifelse(Distance > 5, 5, Distance), fill = Model)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot") + 
  theme_bw()

# ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot_models.jpg', dist_density, dpi = 300, width = 6, height = 4)

# HPI 1
df_agg_rep_HPI1 <- df_agg_rep %>% filter(HPI == 1)
dist_density_HPI1 <- ggplot(df_agg_rep_HPI1, aes(x = ifelse(Distance > 5, 5, Distance), fill = Model)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot (HPI 1)") + 
  theme_bw()

# ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot_HPI1.jpg', dist_density, dpi = 300, width = 6, height = 4)

# HPI 2
df_agg_rep_HPI2 <- df_agg_rep %>% filter(HPI == 2)
dist_density_HPI2 <- ggplot(df_agg_rep_HPI2, aes(x = ifelse(Distance > 5, 5, Distance), fill = Model)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot (HPI 2)") + 
  theme_bw()

# ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot_HPI2.jpg', dist_density, dpi = 300, width = 6, height = 4)

# HPI 3
df_agg_rep_HPI3 <- df_agg_rep %>% filter(HPI == 3)
dist_density_HPI3 <- ggplot(df_agg_rep_HPI3, aes(x = ifelse(Distance > 5, 5, Distance), fill = Model)) +
  geom_density(alpha = 0.5) +  # Adjust alpha for transparency
  labs(x = "Distance (km)", y = "Density") +
  ggtitle("Density Plot (HPI 3)") + 
  theme_bw()

# ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/density_plot_HPI3.jpg', dist_density, dpi = 300, width = 6, height = 4)

combined_figure <- grid.arrange(dist_density_all, dist_density_HPI1, dist_density_HPI2, dist_density_HPI3, nrow = 2)
ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/combined_figure.jpg', combined_figure, dpi = 300, width = 8, height = 4)


################################################################

combined_figure <- grid.arrange(dist_density_blp, dist_density_loglinear, nrow=1)
ggsave('Result/MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/blp_loglinear.jpg', combined_figure, dpi = 300, width = 8, height = 4)





data = read.csv("output/dlm_ground_funchead_coarse.tsv", sep="\t")


library(ggplot2)

plot = ggplot(data, aes(x=Language, y=GrammarLanguage)) + geom_tile(aes(fill=Length))

library(tidyr)
library(dplyr)
dataM = data %>% group_by(Language) %>% summarise(MeanLength = mean(Length), SDLength = sd(Length))
data = merge(data, dataM, by=c("Language"))
plot = ggplot(data, aes(x=Language, y=GrammarLanguage)) + geom_tile(aes(fill=-(Length - MeanLength)/SDLength))


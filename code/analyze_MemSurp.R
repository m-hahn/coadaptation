data = read.csv("output/memsurp_ground_coarse.tsv", sep="\t")


library(ggplot2)

plot = ggplot(data, aes(x=Language, y=GrammarLanguage)) + geom_tile(aes(fill=Length))

library(tidyr)
library(dplyr)
dataM = data %>% group_by(Language) %>% summarise(MeanLength = mean(Length), SDLength = sd(Length))
data = merge(data, dataM, by=c("Language"))
plot = ggplot(data, aes(x=Language, y=GrammarLanguage)) + geom_tile(aes(fill=-(Length - MeanLength)/SDLength))

data$Type = "AllLangs"

plot = ggplot(data, aes(x=Length, fill=Type, color=Type)) + theme_classic() + theme(legend.position="none")   + geom_density(data= data, aes(y=..scaled..))   + geom_bar(data = data %>% filter(as.character(Language) == as.character(GrammarLanguage)) %>% mutate(Type="SameLang") %>% mutate(y=1),  aes(y=y, group=Type), width=0.01, stat="identity", position = position_dodge()) + facet_wrap(~Language, scales="free")




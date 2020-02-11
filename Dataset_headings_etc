setwd("/Users/imyyounge/Documents/4_Masters/4_Machine_learning/Nov_2019_Prescribing_Data")
getwd
# Import data -------------------------------------------------------------
#presc <-read.csv("T201911PDPI BNFT.csv")
toypres <- readRDS("Prescription_data_toy1000.rds")
drug <- read.csv("T201911CHEM SUBS.csv")
gp <- read.csv("T201911ADDR BNFT.csv", header=FALSE)
people <- read.csv("gp-reg-pat-prac-all.csv")
structure <- read.csv("epcmem.csv")
allgp <- read.csv("epraccur.csv", header=FALSE)

#https://digital.nhs.uk/data-and-information/areas-of-interest/prescribing
#More info on how the prescribing data is created

# Sort_out_headings -------------------------------------------------------
colnames(gp) <- c("timepoint", "E8...", "Name", "Address_1", "Address_2", "Address_3", "Area", "Postcode")
colnames(structure) <- c("Organisation_code", "CCG/PCT", "Primary_care_organisation_type", "Join_parent_date", "Left_parent_date", "Amended_record_indicator")
colnames(allgp) <- c("Organisation_code", "Address_1", "National_grouping", 
                     "High_level_health_geography", "Address_2", "Address_3", "Address_4", 
                     "Area", "Postcode", "1974...", "Date_open", "Date_close", "Status_code", 
                     "Subtype", "Commissioner")

# Join datasets together --------------------------------------------------
  # Would it work to do this by postcode? Might be better, and give more accuracy, 
  # cope with different doctors surgeries covering different things
#Think allgp$commissioner same as people$CCG
link <- (merge(allgp, people, by.x = c("Commissioner", "Postcode"), by.y = c("CCG_CODE", "POSTCODE"), all.x = TRUE))
#Maybe gp$E8... same as people$ONSCCGcode this might be the same as toypres$practice
link2 <- (merge(link, gp, by.x = c("Commissioner", "Postcode"), by.y = c("CCG_CODE", "E8..."), all.x = TRUE, all.y=TRUE?))

# Hospital_deaths ---------------------------------------------------------
# Might be an easier dataset to look at
hosp <- read.csv("SHMI_data_at_site_level,_Sep18-Aug19.csv")
dim(hosp)

#Potential question, do they cluster by trust?
summary(toypres)

#Group GP's by postcode
table(people$POSTCODE)
summary(people$POSTCODE)

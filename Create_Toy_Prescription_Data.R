# Toy dataset -------------------------------------------------------------
#Take a random sample of presc so that it is usable in useful time frames
#Aiming to work h=out how to rearrange the dataframes etc
# Function
seed = 100
GenerateToyData=function(x,n=1000,p=NULL,seed=1,observation_consistency=FALSE){
  # x is the original (real) dataset
  # By default (p=NULL), all variables will be kept in the resulting toy example dataset
  # You can choose how many observations to keep in the toy example dataset (no need to go over 100 in general)
  # You can change the seed if you want to generate multiple toy example datasets (not needed in general)
  
  if (is.null(p)){
    p=ncol(x)
  }
  
  if (n>nrow(x)){
    stop(paste0("Please set n to be smaller than the number of rows in the original dataset (i.e. n<=", nrow(x), ")"))
  }
  
  # Random sub-sample of rows and columns
  set.seed(seed)
  s=sample(nrow(x), size=n)
  if (p==ncol(x)){
    xtoy=x[s,1:p]
  } else {
    xtoy=x[s,sample(p)]
  }
  if (!observation_consistency){
    # Permutation of the observations by variable (keeping the structure of missing data)
    for (k in 1:p){
      xtmp=xtoy[,k]
      xtmp[!is.na(xtmp)]=sample(xtmp[!is.na(xtmp)])
      xtoy[,k]=xtmp
    }
  }
  rownames(xtoy)=1:nrow(xtoy)
  return(xtoy)
}

# Generate a toy example dataset with all variables and a subset of 1000 observations
toypres <- GenerateToyData(x=presc, seed=1)
saveRDS(toypres, "Prescription_data_toy1000.rds")
write.csv(toypres,'ToyPres.csv')
write.csv(people, "people.csv")
write.csv(structure, "structure.csv")


## about: convert original Add Health CD Data from .paj format --> .gml format
## note: The attribute field for "totalnoms" will be >= the out-degree in the school network
## because according to the Feb. 2005/Codebook "Students could list friends who did not attend the same school or
## sister school by using a special code. These included (a) friends who went to the same school
## but who were not in the roster, (b) friends who attended the sister school but were not in the roster,
## and (c) friends who did not attend either school. These “out-of-school” nominations cannot be linked
## to other students in the school, and are thus treated as missing data here.
## Out-of- school friends are included in the total number of nominations made by each student."

rm(list=ls())

## note: user sets path to file location of raw add health data
file_path_to_raw_add_health_data <-"/Users/kristen/Dropbox/gender_graph_data/add-health/cd_data/structure_nocontract/"
setwd(file_path_to_raw_add_health_data)

library(intergraph)
library(network)
library(igraph)


for(files in list.files()){
  print(files)
  file_num <- as.numeric(gsub(".paj","",gsub("comm", "", files))) ## get file name number
  comm <- read.paj(files) ## read in file

  ## first get node attributes
  attributes <- c(names(comm$partitions)) # get names of all possible attributes which may vary

  # create data frame of vertex id + attributes
  attribute_df <- data.frame("vertex.names" =( c(network::get.vertex.attribute(comm$networks[[1]],
                                                                      'vertex.names'))))
  ## check for duplicate node IDs
  if(anyDuplicated(c(as.integer( c(network::get.vertex.attribute(comm$networks[[1]], 'vertex.names'))))) > 0){
    print('error: duplicated node IDs')
  }

  num <- nrow(attribute_df)
  for(i in 1:length(attributes)){
    if(length(c(comm$partitions[i])[[1]]) != num){
      print('mismatch length error')
    }
    # note: implicitly assuming order of attributes corresponds with node id order
    attribute_df[,attributes[i]] <- c(comm$partitions[i])
  }
  ## kristen - 7/27/2016 - spot-checked comm1 attribute aligns with raw file
  ## refs: https://cran.r-project.org/web/packages/intergraph/vignettes/howto.html 
  ## in particular the "Handling attributes" section since igraph/network store them differently
  

  g <- intergraph::asIgraph(comm$networks[[1]])

  ## attach node attribute data.frame to graph object g
  for(j in 1:length(attributes)){
    g <- igraph::set.vertex.attribute(g,
                  index = as.integer(as.character(attribute_df[,c('vertex.names')])), # since vertex.names is a factor object in R - we have to first convert to character type, and then to an integer
                  name = attributes[j],
                  value=c(attribute_df[,c(attributes[j])]))
  }
  g <- igraph::delete_vertex_attr(g, 'na')
  g <- igraph::delete_edge_attr(g, 'na')
  write.graph(g, file = paste0('/Users/kristen/Desktop/converted_gml/', gsub(".paj", ".gml", files)),format = c('gml')) ## user sets path to location of converted files
  #write.graph(g, file = paste0('../../converted_gml/', gsub(".paj", ".gml", files)),format = c('gml')) ## user sets path to location of converted files
}

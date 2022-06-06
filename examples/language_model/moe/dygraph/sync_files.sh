#!/bin/bash                                                                                                                                               
                                                                                                                                                            
# get sshd port                                                                                                                                           
sshport=$(lsof -i | grep sshd | awk '{print $9}' | sed s/\*://)                                                                                           
                                                                                                                                                            
hostfile=${TRAIN_WORKSPACE}/hostfile                                                                                                                      
hostlist=$(cat $hostfile | awk '{print $1}' | xargs)                                                                                                      
for host in ${hostlist[@]}; do                                                                                                                            
  #ssh $host "ls $PWD"                                                                                                                                    
  echo "scp $1 to $host"                                                                                                                                  
  scp -r $1 ${host}:${PWD}                                                                                                                                
done

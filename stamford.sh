GRAPH=/home/ww/stamford-hill/data/stamford.graphml
MODEL=/home/ww/stamford-hill/stamford-places.hka
ABC=sqlite:////home/ww/stamford-hill/net.db
GEN=1

time netabc -l 90 -f beta_s 0.12 -f beta_y 0.22 -f beta_m 0.22 -f beta_g 0.24 \
     abc -d ${ABC} \
	   graph -f ${GRAPH} \
     stamford \
	   netkappa -f ${MODEL} \
     fit -e 0.0 -s 500 -m L2 \
         -r DHSAR 0.0 \
		     -g beta_a 0.14 0.014 -g beta_c 0.14 0.014

exit 0

#         -r HSAR 0.60 -r GSAR 0.67 -r SSAR 0.5 -r YSAR 0.74 -r MSAR 0.74 \
#		     -p beta_h 0.1 0.2 -p beta_g 0.1 0.2 -p beta_s 0.1 0.2 -p beta_y 0.1 0.2 -p beta_m 0.1 0.2 \

time netabc -l 120 -f beta_h 0.14 -f beta_s 0.12 -f beta_y 0.22 -f beta_m 0.22 -f beta_g 0.24 \
     store -d net.h5 -p net \
     graph -f ${GRAPH} \
     stamford \
     netkappa -f /home/ww/stamford-hill/stamford-places.hka \
     run -n 128 plot -o S,E,I,R -o HSAR,GSAR,SSAR,YSAR,MSAR -o DHSAR,DGSAR,DSSAR,DYSAR,DMSAR -o ACTH,ACTG,ACTS,ACTY,ACTM -o Fi,Mi -o Fr,Mr

#netabc abc -d ${ABC} -g ${GEN} plot_abc -o net -p beta_h,beta_g,beta_s,beta_y,beta_m
#netabc abc -d ${ABC} -g ${GEN} plot_abc -o net -g -p beta_h -p beta_g -p beta_s -p beta_y -p beta_m
#netabc abc -d ${ABC} -g ${GEN} plot_abc -o net -p beta_g,beta_m
#netabc abc -d ${ABC} -g ${GEN} plot_abc -o net -g -p beta_g -p beta_m

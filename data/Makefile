GRAPH=stamford.graphml
MODEL=../kappa/stamford-places.hka
ABCDB=stamford.db
UABCDB=stamford-uniform.db
TRAJS=stamford.h5
SAMPLES=256
POPS?=8

all:
	make ${ABCDB}
	make ${TRAJS}
	make plots
	make SCALE=1.2 sensitivity sensitivity_plots || true
	make SCALE=1.1 sensitivity sensitivity_plots || true
	make SCALE=1.05 sensitivity sensitivity_plots || true
	make SCALE=0.99 sensitivity sensitivity_plots || true
	make SCALE=0.98 sensitivity sensitivity_plots || true
	make SCALE=0.95	sensitivity sensitivity_plots || true
	make SCALE=0.9 sensitivity sensitivity_plots || true
	make SCALE=0.8 sensitivity sensitivity_plots || true
	make SCALE=0.7 sensitivity sensitivity_plots || true

plots: rule-activities.png wasserstein.png censoring.png

${GRAPH}: stamford_hill_survey.zip combined_data.csv
	stamford_graph $^ > $@

${UABCDB}: ${GRAPH} ${MODEL}
	netabc -l 90 \
		abc -d sqlite:///${UABCDB} \
		graph -f ${GRAPH} \
		stamford \
		netkappa -f ${MODEL} \
		fit -n ${POPS} -e 0.0 -s 512 -m L2 -r DHSAR 0.0 ${RESUME} \
			-u beta_e 0.0 0.5 \
			-u beta_h 0.0 0.5 \
			-u beta_g 0.0 0.5 \
			-u beta_s 0.0 0.5 \
			-u beta_y 0.0 0.5 \
			-u beta_m 0.0 0.5

${ABCDB}: ${UABCDB}
	netabc -l 90 \
		abc -d sqlite:///${ABCDB} \
		graph -f ${GRAPH} \
		stamford \
		netkappa -f ${MODEL} \
		fit -n ${POPS} -e 0.0 -s 512 -m L2 -r DHSAR 0.0 ${RESUME} \
		`netabc abc -d sqlite:///${UABCDB} | awk '{ printf(" -g %s %f %f", $$4, $$6, $$6/10); }'`

${TRAJS}: ${ABCDB}
	time netabc -l 90 \
		`netabc abc -d sqlite:///${UABCDB} | awk '{ printf(" -d %s %f %f", $$4, $$6, $$6/10); }'` \
     store -d ${TRAJS} -p net \
     graph -f ${GRAPH} \
     stamford \
     netkappa -f ${MODEL} -s snapshots/stamford-snapshot \
     run -n 256

sensitivity: sensitivity-${SCALE}.h5
sensitivity-${SCALE}.h5: ${GRAPH} ${MODEL} params.txt
	time netabc -l 90 \
		`netabc abc -d sqlite:///${UABCDB} | awk '{ printf(" -d %s %f %f", $$4, $$6, $$6/10); }'` \
     store -d $@ -p net \
     graph -f ${GRAPH} \
     stamford -s ${SCALE} \
     netkappa -f ${MODEL} -s snapshots/sensitivity-${SCALE}-snapshot \
     run -n 256

net-final-activity.png: ${TRAJS}
	netabc store -d $< -p net graph -f ${GRAPH} plot_scaled_activity \
		-o ACTH household \
		-o ACTS school \
		-o ACTY yeshiva \
		-o ACTG synagogue \
		-o ACTM mikvah

rule-activities.png: ${TRAJS}
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford plot_stamford_act

wasserstein.png: ${TRAJS}
	netabc graph -f ${GRAPH} stamford plot_stamford_wass snapshots/stamford-snapshot-*.graphml

censoring.png: ${GRAPH}
	netabc graph -f ${GRAPH} stamford plot_stamford_cens

sensitivity_plots: rule-activities-${SCALE}.png wasserstein-${SCALE}.png
rule-activities-${SCALE}.png: sensitivity-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -s ${SCALE} plot_stamford_act -o $@
wasserstein-${SCALE}.png: sensitivity-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -s ${SCALE} plot_stamford_wass -o $@ snapshots/sensitivity-${SCALE}-snapshot-*.graphml

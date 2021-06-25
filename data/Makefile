GRAPH=stamford.graphml
MODEL=../kappa/stamford-places.hka
ABCDB=stamford.db
UABCDB=stamford-uniform.db
TRAJS=stamford.h5
SAMPLES=1024
POPS?=8

## susceptibility of children
s_preschool=0.25
s_primary=0.5
s_secondary=0.75

all:
	make plots
	make data
	make SCALE=1.2 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=1.1 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=1.05 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=0.99 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=0.98 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=0.95	sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=0.9 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=0.8 sensitivity sensitivity_plots sensitivity_data || true
	make SCALE=0.7 sensitivity sensitivity_plots sensitivity_data || true
#	make SCALE=0.9 PLACE=synagogue close close_plots close_data || true
	make PCTILE=0.9 split split_plots split_data || true
	make PCTILE=0.8 split split_plots split_data || true
	make PCTILE=0.7 split split_plots split_data || true
	make PCTILE=0.6 split split_plots split_data || true
	make PCTILE=0.5 split split_plots split_data || true

plots: censoring.png test-distributions.png rule-activities.png wasserstein.png introductions.png epidemic-S-E-I-R.png rule-activities-tight.png wasserstein-tight.png introductions-tight.png epidemic-tight-S-E-I-R.png
data: introductions.tsv introductions-tight.tsv multi.tsv multi-tight.tsv rule-activities.tsv rule-activities-summary.tsv rule-activities-tight.tsv rule-activities-summary-tight.tsv wasserstein.tsv seropositivity.tsv wasserstein-tight.tsv seropositivity-tight.tsv age-distributions.tsv

${GRAPH}: stamford_hill_survey.zip combined_data.csv
	stamford_graph $^ > $@
minimal-${GRAPH}: stamford_hill_survey.zip combined_data.csv
	stamford_graph $^ --minimal > $@

${UABCDB}: ${GRAPH} ${MODEL}
	netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${UABCDB} \
		graph -f ${GRAPH} \
		stamford \
		netkappa -f ${MODEL} \
		fit -n ${POPS} -e 0.0 -s 512 -m L2 -r DHSAR 0.0 ${RESUME} \
			-u beta_e 0.0 0.5 \
			-u beta_h 0.0 0.5 \
			-u beta_p 0.0 0.5 \
			-u beta_s 0.0 0.5 \
			-u beta_g 0.0 0.5 \
			-u beta_m 0.0 0.5

${ABCDB}: ${UABCDB}
	netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${ABCDB} \
		graph -f ${GRAPH} \
		stamford \
		netkappa -f ${MODEL} \
		fit -n ${POPS} -e 0.05 -s 512 -m L2 -r Dist 0.0 ${RESUME} \
		-g beta_f 0.07 0.02 \
		`netabc abc -i -d sqlite:///${UABCDB} | awk -F= '/^[a-z]/ { v=$$1 } /mean=/ { printf(" -g %s %f %f", v, $$2, $$2/10); }'`

split_fit: split-${PCTILE}-${ABCDB}
split-${PCTILE}-${ABCDB}: ${ABCDB}
	netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///split-${PCTILE}-${ABCDB} \
		graph -f ${GRAPH} \
		stamford -x synagogue ${PCTILE} -x primary ${PCTILE} \
		netkappa -f ${MODEL} \
		fit -n ${POPS} -e 0.05 -s 512 -m L2 -r DHSAR 0.0 ${RESUME} \
		`netabc abc -i -d sqlite:///${ABCDB} | awk -F= '/^[a-z]/ { v=$$1 } /mean=/ { printf(" -g %s %f %f", v, $$2, $$2/10); }'`

${TRAJS}: ${ABCDB}
	rm -f snapshots/stamford-snapshot-*
	time netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${ABCDB} \
		store -d ${TRAJS} -p net \
		graph -f ${GRAPH} \
		stamford \
		netkappa -f ${MODEL} -s snapshots/stamford-snapshot \
		run -n ${SAMPLES}

tight.h5: ${ABCDB}
	rm -f snapshots/tight-snapshot*
	time netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${ABCDB} -f \
		store -d $@ -p net \
		graph -f ${GRAPH} \
		stamford \
		netkappa -f ${MODEL} -s snapshots/tight-snapshot \
		run -n ${SAMPLES}

sensitivity: sensitivity-${SCALE}.h5
sensitivity-${SCALE}.h5: ${GRAPH} ${MODEL} ${ABCDB}
	rm -f snapshots/sensitivity-${SCALE}-snapshot*
	time netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${ABCDB} \
		store -d $@ -p net \
		graph -f ${GRAPH} \
		stamford -s ${SCALE} \
		netkappa -f ${MODEL} -s snapshots/sensitivity-${SCALE}-snapshot \
		run -n ${SAMPLES}

close: close-${PLACE}-${SCALE}.h5
close-${PLACE}-${SCALE}.h5: ${GRAPH} ${MODEL} ${ABCDB}
	rm -f snapshots/close-${PLACE}-${SCALE}-snapshot*
	time netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${ABCDB} \
		store -d $@ -p net \
		graph -f ${GRAPH} \
		stamford -c ${PLACE} ${SCALE} \
		netkappa -f ${MODEL} -s snapshots/close-${PLACE}-${SCALE}-snapshot \
		run -n ${SAMPLES}

split: split-${PCTILE}.h5
split-${PCTILE}.h5: ${GRAPH} ${MODEL} ${ABCDB}
	rm -f snapshots/split-${PCTILE}-snapshot*
	time netabc -l 90 \
		-f s_preschool ${s_preschool} \
		-f s_primary ${s_primary} \
		-f s_secondary ${s_secondary} \
		abc -d sqlite:///${ABCDB} \
		store -d $@ -p net \
		graph -f ${GRAPH} \
		stamford -x synagogue ${PCTILE} -x primary ${PCTILE} \
		netkappa -f ${MODEL} -s snapshots/split-${PCTILE}-snapshot \
		run -n ${SAMPLES}

net-final-activity.png: ${TRAJS}
	netabc store -d $< -p net graph -f ${GRAPH} plot_scaled_activity \
		-o ACTH household \
		-o ACTP primary \
		-o ACTS secondary \
		-o ACTG synagogue \
		-o ACTM mikvah \
		-o ACTE environment

rule-activities.png: ${TRAJS}
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford plot_stamford_act -o $@
rule-activities.tsv: ${TRAJS}
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford write_stamford_act -o $@
rule-activities-summary.tsv: ${TRAJS}
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford write_stamford_act -o $@ -s

wasserstein.png: ${TRAJS}
	netabc graph -f ${GRAPH} stamford plot_stamford_wass -o $@ snapshots/stamford-snapshot-*.graphml
wasserstein.tsv: ${TRAJS}
	netabc graph -f ${GRAPH} stamford write_stamford_wass -o $@ snapshots/stamford-snapshot-*.graphml

seropositivity.tsv: ${TRAJS}
	netabc graph -f ${GRAPH} stamford write_stamford_sero -o $@ snapshots/stamford-snapshot-*.graphml

introductions.png: ${TRAJS}
	netabc plot_stamford_intro -o $@ snapshots/stamford-snapshot*.graphml
introductions.tsv: ${TRAJS}
	netabc write_stamford_intro -o $@ snapshots/stamford-snapshot*.graphml
multi.tsv: ${TRAJS}
	netabc write_stamford_multi -o $@ snapshots/stamford-snapshot*.graphml

epidemic-S-E-I-R.png: ${TRAJS}
	netabc -l 90 store -r -d $< -p net plot -o S,E,I,R --output epidemic

test-distributions.png: ${GRAPH}
	netabc graph -f ${GRAPH} stamford plot_stamford_wass -o $@

censoring.png: ${GRAPH}
	netabc graph -f ${GRAPH} stamford plot_stamford_cens

rule-activities-tight.png: tight.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford plot_stamford_act -o $@
rule-activities-tight.tsv: tight.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford write_stamford_act -o $@
rule-activities-summary-tight.tsv: tight.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford write_stamford_act -o $@ -s

wasserstein-tight.png: tight.h5
	netabc graph -f ${GRAPH} stamford plot_stamford_wass -o $@ snapshots/tight-snapshot-*.graphml
wasserstein-tight.tsv: tight.h5
	netabc graph -f ${GRAPH} stamford write_stamford_wass -o $@ snapshots/tight-snapshot-*.graphml

seropositivity-tight.tsv: ${TRAJS}
	netabc graph -f ${GRAPH} stamford write_stamford_sero -o $@ snapshots/tight-snapshot-*.graphml

introductions-tight.png: tight.h5
	netabc plot_stamford_intro -o $@ snapshots/tight-snapshot*.graphml
introductions-tight.tsv: tight.h5
	netabc write_stamford_intro -o $@ snapshots/tight-snapshot*.graphml
multi-tight.tsv: tight.h5
	netabc write_stamford_multi -o $@ snapshots/tight-snapshot*.graphml

epidemic-tight-S-E-I-R.png: tight.h5
	netabc -l 90 store -r -d $< -p net plot -o S,E,I,R --output epidemic-tight

sensitivity_plots: rule-activities-${SCALE}.png wasserstein-${SCALE}.png introductions-${SCALE}.png epidemic-sensitivity-${SCALE}-S-E-I-R.png
rule-activities-${SCALE}.png: sensitivity-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -s ${SCALE} plot_stamford_act -o $@
wasserstein-${SCALE}.png: sensitivity-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -s ${SCALE} plot_stamford_wass -o $@ snapshots/sensitivity-${SCALE}-snapshot-*.graphml
introductions-${SCALE}.png: sensitivity-${SCALE}.h5
	netabc plot_stamford_intro -o $@ snapshots/sensitivity-${SCALE}-snapshot*.graphml
epidemic-sensitivity-${SCALE}-S-E-I-R.png: sensitivity-${SCALE}.h5
	netabc -l 90 store -r -d $< -p net plot -o S,E,I,R --output epidemic-sensitivity-${SCALE}

sensitivity_data: introductions-${SCALE}.tsv multi-${SCALE}.tsv wasserstein-${SCALE}.tsv seropositivity-${SCALE}.tsv rule-activities-${SCALE}.tsv rule-activities-summary-${SCALE}.tsv
introductions-${SCALE}.tsv: sensitivity-${SCALE}.h5
	netabc write_stamford_intro -o $@ snapshots/sensitivity-${SCALE}-snapshot*.graphml
multi-${SCALE}.tsv: sensitivity-${SCALE}.h5
	netabc write_stamford_multi -o $@ snapshots/sensitivity-${SCALE}-snapshot*.graphml
wasserstein-${SCALE}.tsv: sensitivity-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -s ${SCALE} write_stamford_wass -o $@ snapshots/sensitivity-${SCALE}-snapshot*.graphml
seropositivity-${SCALE}.tsv: sensitivity-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -s ${SCALE} write_stamford_sero -o $@ snapshots/sensitivity-${SCALE}-snapshot*.graphml
rule-activities-${SCALE}.tsv: sensitivity-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -s ${SCALE} write_stamford_act -o $@
rule-activities-summary-${SCALE}.tsv: sensitivity-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -s ${SCALE} write_stamford_act -o $@ -s

close_plots: rule-activities-${PLACE}-${SCALE}.png wasserstein-${PLACE}-${SCALE}.png introductions-close-${PLACE}-${SCALE}.png epidemic-close-${PLACE}-${SCALE}-S-E-I-R.png
rule-activities-${PLACE}-${SCALE}.png: close-${PLACE}-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -c ${PLACE} ${SCALE} plot_stamford_act -o $@
wasserstein-${PLACE}-${SCALE}.png: close-${PLACE}-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -c ${PLACE} ${SCALE} plot_stamford_wass -o $@ snapshots/close-${PLACE}-${SCALE}-snapshot-*.graphml
introductions-close-${PLACE}-${SCALE}.png: close-${PLACE}-${SCALE}.h5
	netabc plot_stamford_intro -o $@ snapshots/close-${PLACE}-${SCALE}-snapshot*.graphml
epidemic-close-${PLACE}-${SCALE}-S-E-I-R.png: close-${PLACE}-${SCALE}.h5
	netabc -l 90 store -r -d $< -p net plot -o S,E,I,R --output epidemic-close-${PLACE}-${SCALE}

close_data: introductions-close-${PLACE}-${SCALE}.tsv multi-close-${PLACE}-${SCALE}.tsv rule-activities-${PLACE}-${SCALE}.tsv rule-activities-summary-${PLACE}-${SCALE}.tsv  wasserstein-close-${PLACE}-${SCALE}.tsv seropositivity-close-${PLACE}-${SCALE}.tsv
introductions-close-${PLACE}-${SCALE}.tsv: close-${PLACE}-${SCALE}.h5
	netabc write_stamford_intro -o $@ snapshots/close-${PLACE}-${SCALE}-snapshot*.graphml
multi-close-${PLACE}-${SCALE}.tsv: close-${PLACE}-${SCALE}.h5
	netabc write_stamford_multi -o $@ snapshots/close-${PLACE}-${SCALE}-snapshot*.graphml
rule-activities-${PLACE}-${SCALE}.tsv: close-${PLACE}-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -c ${PLACE} ${SCALE} write_stamford_act -o $@
rule-activities-summary-${PLACE}-${SCALE}.tsv: close-${PLACE}-${SCALE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -c ${PLACE} ${SCALE} write_stamford_act -o $@ -s
wasserstein-${PLACE}-${SCALE}.tsv: close-${PLACE}-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -c ${PLACE} ${SCALE} write_stamford_wass -o $@ snapshots/close-${PLACE}-${SCALE}-snapshot-*.graphml
seropositivity-${PLACE}-${SCALE}.tsv: close-${PLACE}-${SCALE}.h5
	netabc graph -f ${GRAPH} stamford -c ${PLACE} ${SCALE} write_stamford_sero -o $@ snapshots/close-${PLACE}-${SCALE}-snapshot-*.graphml

split_plots: rule-activities-split-${PCTILE}.png wasserstein-split-${PCTILE}.png introductions-split-${PCTILE}.png epidemic-split-${PCTILE}-S-E-I-R.png #split-${PCTILE}-kde.png
rule-activities-split-${PCTILE}.png: split-${PCTILE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -x synagogue ${PCTILE} -x primary ${PCTILE} plot_stamford_act -o $@
wasserstein-split-${PCTILE}.png: split-${PCTILE}.h5
	netabc graph -f ${GRAPH} stamford -x synagogue ${PCTILE} -x primary ${PCTILE} plot_stamford_wass -o $@ snapshots/split-${PCTILE}-snapshot-*.graphml
introductions-split-${PCTILE}.png: split-${PCTILE}.h5
	netabc plot_stamford_intro -o $@ snapshots/split-${PCTILE}-snapshot*.graphml
epidemic-split-${PCTILE}-S-E-I-R.png: split-${PCTILE}.h5
	netabc -l 90 store -r -d $< -p net plot -o S,E,I,R --output epidemic-split-${PCTILE}

split_data: introductions-split-${PCTILE}.tsv multi-split-${PCTILE}.tsv rule-activities-split-${PCTILE}.tsv rule-activities-summary-split-${PCTILE}.tsv wasserstein-split-${PCTILE}.tsv seropositivity-split-${PCTILE}.tsv
introductions-split-${PCTILE}.tsv: split-${PCTILE}.h5
	netabc write_stamford_intro -o $@ snapshots/split-${PCTILE}-snapshot*.graphml
multi-split-${PCTILE}.tsv: split-${PCTILE}.h5
	netabc write_stamford_multi -o $@ snapshots/split-${PCTILE}-snapshot*.graphml
rule-activities-split-${PCTILE}.tsv: split-${PCTILE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -x synagogue ${PCTILE} -x primary ${PCTILE} write_stamford_act -o $@
rule-activities-summary-split-${PCTILE}.tsv: split-${PCTILE}.h5
	netabc store -r -d $< -p net graph -f ${GRAPH} stamford -x synagogue ${PCTILE} -x primary ${PCTILE} write_stamford_act -o $@ -s
wasserstein-split-${PCTILE}.tsv: split-${PCTILE}.h5
	netabc graph -f ${GRAPH} stamford -x synagogue ${PCTILE} -x primary ${PCTILE} write_stamford_wass -o $@ snapshots/split-${PCTILE}-snapshot-*.graphml
seropositivity-split-${PCTILE}.tsv: split-${PCTILE}.h5
	netabc graph -f ${GRAPH} stamford -x synagogue ${PCTILE} -x primary ${PCTILE} write_stamford_sero -o $@ snapshots/split-${PCTILE}-snapshot-*.graphml

split_distances.tsv:
	rm -f $@
	(echo -en '1.0\t'; netabc graph -f ${GRAPH} stamford compute_stamford_dist snapshots/stamford-snapshot-*.graphml) >> $@
	(echo -en '0.9\t'; netabc graph -f ${GRAPH} stamford -x synagogue 0.9 -x primary 0.9 compute_stamford_dist snapshots/split-0.9-*.graphml) >> $@
	(echo -en '0.8\t'; netabc graph -f ${GRAPH} stamford -x synagogue 0.8 -x primary 0.8 compute_stamford_dist snapshots/split-0.8-*.graphml) >> $@
	(echo -en '0.7\t'; netabc graph -f ${GRAPH} stamford -x synagogue 0.7 -x primary 0.7 compute_stamford_dist snapshots/split-0.7-*.graphml) >> $@
	(echo -en '0.6\t'; netabc graph -f ${GRAPH} stamford -x synagogue 0.6 -x primary 0.6 compute_stamford_dist snapshots/split-0.6-*.graphml) >> $@
	(echo -en '0.5\t'; netabc graph -f ${GRAPH} stamford -x synagogue 0.5 -x primary 0.5 compute_stamford_dist snapshots/split-0.5-*.graphml) >> $@

uniform-kde.png: ${UABCDB}
	netabc abc -d sqlite:///${UABCDB} plot_kde -a -o uniform
stamford-kde.png: ${ABCDB}
	netabc abc -d sqlite:///${ABCDB} plot_kde -a -o stamford
split-${PCTILE}-kde.png: ${ABCDB}
	netabc abc -d sqlite:///split-${PCTILE}-${ABCDB} plot_kde -a -o split-${PCTILE}

data_plots:
	netabc graph -f ${GRAPH} stamford plot_stamford_data -e -o stamford-enriched
	netabc graph -f ${GRAPH} stamford plot_stamford_data -r -o stamford-random
	netabc graph -f ${GRAPH} stamford plot_stamford_data -e -s -o stamford-enriched-serology
	netabc graph -f ${GRAPH} stamford plot_stamford_data -r -s -o stamford-random-serology

age-distributions.tsv: ${GRAPH}
	netabc graph -f ${GRAPH} stamford write_stamford_demo -o distributions.tsv

seropositivity-empirical.tsv: ${GRAPH}
	netabc write_stamford_sero -e -o $@ $^

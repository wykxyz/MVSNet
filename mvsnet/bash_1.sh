tanks_dir='/home/haibao637/data/tankandtemples/intermediate//'
test=("Family" "Horse" "Lighthouse" "Fransics" "M60" "Panther" "Playground" "Train" ) 
for p in ${test[@]}
do


	test=$tanks_dir/$p
	echo $test
	python test_1.py --dense_folder=$test --max_w=960 --max_h=520 ##--max_d=512
	#python postprocess.py --dense_folder=$test
	python depthfusion.py --dense_folder=$test --prob_threshold=0.15

done

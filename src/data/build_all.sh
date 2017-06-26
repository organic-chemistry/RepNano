
list=( ( "sub_template.InDeepNano" "substituted" ) ( "" "test" ) )

for element in "${list[@]}"
do
        file=${element[0]}
        dir=${element[1]}
        echo file dir

      #  python split_training.py $root/$element $external/S288C_reference_sequence_R64-2-1_20150113.fa
        #python prepare_dataset.py temp $root/$element.train ../ForJM/control/ ../ForJM/control_template_train

done


#
# python split_training.py  ../ForJM/sub_template.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
# python split_training.py  ../ForJM/control_template.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
#
# python prepare_dataset.py temp ../ForJM/control_template.InDeepNano.train ../ForJM/control/ ../ForJM/control_template_train
# python prepare_dataset.py temp ../ForJM/control_template.InDeepNano.test ../ForJM/control/ ../ForJM/control_template_test
# python prepare_dataset.py temp ../ForJM/sub_template.InDeepNano.test ../ForJM/substituted/ ../ForJM/sub_template_test
# python prepare_dataset.py temp ../ForJM/sub_template.InDeepNano.train ../ForJM/substituted/ ../ForJM/sub_template_train
#
#
# python split_training.py  ../ForJM/control_complement.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
# python split_training.py  ../ForJM/sub_complement.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
#
#
# mkdir ../ForJM/sub_complement_train
# mkdir ../ForJM/sub_complement_test
# mkdir ../ForJM/control_complement_test
# mkdir ../ForJM/control_complement_train
#
# python prepare_dataset.py comp ../ForJM/sub_complement.InDeepNano.train ../ForJM/substituted/ ../ForJM/sub_complement_train
# python prepare_dataset.py comp ../ForJM/sub_complement.InDeepNano.test ../ForJM/substituted/ ../ForJM/sub_complement_test
# python prepare_dataset.py comp ../ForJM/control_complement.InDeepNano.test ../ForJM/control/ ../ForJM/control_complement_test
# python prepare_dataset.py comp ../ForJM/control_complement.InDeepNano.train ../ForJM/control/ ../ForJM/control_complement_train

# DDPG-Torcs-Tensorflow
Implementation of Deep Deterministic Policy Gradient Algorithm in Tensorflow to play Torcs 

To train or Test 
Modify: my_config.py
run: python playGame_tensorflow.py

Usually you should observe high rewards for around ~300th or 400th episode. If you are still getting low rewards. Please restart training. 

For faster training avoid the complete rendering of the graphics. This can be done by replacing the practice.xml to practice_results_only.xml

There is a configuration file "practice.xml" that exists in one of the 2 folders which can be found by running the command

sudo find / -name practice.xml

One of the folders is "~/.torcs/config/raceman"
Other can be found as specified.

Torcs can run in 2 modes either the complete race simulation or results only <10X faster>
For training I replace practice.xml by practice_results_mode.xml
For visualization again replace practice.xml by practice_normal_mode.xml


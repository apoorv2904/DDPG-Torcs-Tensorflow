echo $1


echo 'Termination Causes : '
echo
grep 'Terminating because' $1 | tail -10

echo 
echo 'All Rewards '
echo 
grep 'TOTAL REWARD .* Reward' $1 | tail -10

echo
echo 'Rewards > 1000 '
echo
grep 'TOTAL REWARD .* Reward [1-9]...\.' $1 | tail -10


echo
echo 'Rewards > 10000 '
echo
grep 'TOTAL REWARD .* Reward [1-9]....\.' $1 | tail -$2


echo
echo 'Rewards > 100000 '
echo
grep 'TOTAL REWARD .* Reward [1-9].....\.' $1 | tail -$2

echo
echo 'Early Stoppings  '
echo
grep 'Early Stopping: 1' $1 | tail -$2

echo
echo 'Any Nans  '
echo
grep 'nan' $1

echo
echo 'Current Epsilon '
echo
tail -3 $1 | grep 'Epsilon'

echo
echo 'Model Last Saved '
echo
grep 'Now we save model' $1 | tail -2

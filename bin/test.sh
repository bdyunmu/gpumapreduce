s=$(cat output|awk -F':' '{print $3}'|awk -F' ' '{print $1}')
t=0;
for i in $s
do
t=$(($t+$i))
done
echo $t

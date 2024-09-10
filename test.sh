current_dir=$(pwd)
INPUT="$current_dir/input"
OUTPUT="$current_dir/output"
SUBMIT="$current_dir/submit"

touch logFile

echo "======= START ========="
echo "======= START =========" >> $current_dir/logFile
date >> logFile

cd $SUBMIT
ROLLNO=$(ls *.cu | tail -1 | cut -d'.' -f1)
echo "$ROLLNO"
cp ${ROLLNO}.cu main.cu

bash compile.sh
wait
for testcase in $INPUT/*;
do
	filename=${testcase##*/}
	./main.out $INPUT/$filename output.txt >> $current_dir/logFile
	diff $OUTPUT/$filename output.txt -b > /dev/null 2>&1
	exit_code=$?

	if (($exit_code==0)); then
		echo "$filename success"
		echo "$filename success" >> $current_dir/logFile
	else
		echo "$filename failure"
		echo "$filename failure" >> $current_dir/logFile
	fi
done
echo "========== END ========="
echo "========== END =========" >> $current_dir/logFile

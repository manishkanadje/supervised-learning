---------
Read Me
---------
1. trainMLP.py
   Execution - python trainMLP.py train_data.csv

2. executeMLP.py
   Execution - python executeMLP.py test_data.csv weighfile
   
   weightfile for executeMLP.py are
   1. weights0.csv
   2. weights10.csv
   3. weights100.csv
   4. weights1000.csv
   5. weights10000.csv

   Number in the file name represent the number of epochs

3. trainDT.py
   Execution - python trainDT.py train_data.csv

4. executeDT.py
   Execution - python executeDT.py treefile

   treefile for executeDT.py are
   1. tree1.p
   2. tree10.p
   3. tree100.p
   4. tree1000.p

---------
Results
---------
1. trainMLP.py create a graph of SSE vs number of epochs
   It also creates .csv files containing weights of the MLP after certain iteration.
   It creates .p pickle files for some object. Please ignore those files

2. executeMLP.py prints correct and incorrect classification, recognition rate, 
   confusion matrix and final profit in the standard output.
   It creates an image displaying the classification regions using given weightfile

3. trainDT.py produces a graph of SSE vs number of trees in the forest.
   It creates .p files mentioned above containing trees in the forest

2. executeDT.py prints correct and incorrect classification, recognition rate, 
   confusion matrix and final profit in the standard output.
   It creates an image displaying the classification regions using given treefile

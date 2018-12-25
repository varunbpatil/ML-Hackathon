For more detailed information, see my blog post on this Hackathon -

https://varunbpatil.github.io/2018/12/25/genpact-ml-hackathon.html



==================================
Running the code file - genpact.py
==================================

1. Place the 'genpact.py' file in the same directory that contains the
   training data 'train_GzS76OK' and the test data 'test_QoiMO9B.csv'.

   The directory structure should look like this:

   ~/genpact$ tree
   .
   ___ genpact.py
   ___ test_QoiMO9B.csv
   ___ train_GzS76OK
       ___ fulfilment_center_info.csv
       ___ meal_info.csv
       ___ train.csv


2. Run the python file - 'genpact.py' as below.

   $ python genpact.py


3. A 'submission.csv' file is created in the same directory containing
   the submission to be made for this contest corresponding to the
   test data 'test_QoiMO9B.csv'.



=============
Dependencies:
=============
1. Python 3.6.5
2. Pandas 0.23.0
3. Numpy 1.14.3
4. scikit-learn 0.19.1
5. LightGBM 2.2.1

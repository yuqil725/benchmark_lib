![image](./arch.png)

# Benchmark data generation
Data generation can be done in this notebook
https://colab.research.google.com/drive/15xCj06CZuHAxSskpSFGAtom2iEZzfdlE?usp=sharing

Basically the code would run selected model configurations for 5 epoches, and then calculate the average training time of the last 4 epochs, and saves the model to google drive.

After, you need to
1. Download the saved model from Google Drive to local
2. Upload the zip to s3 `benchmark-model` bucket
3. Modify the access to public


# Use Benchmark Library
After the benchmark data you like has been generated, you should do the following
1. Add benchmark data configuration to `benchmark.py` file
2. Push the changes the github
3. Follow the steps in https://colab.research.google.com/drive/14SNptCObXXkldps3a6pIsqUnb0EdlFLk?usp=sharing

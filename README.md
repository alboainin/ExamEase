# **ExamEase**

### **Description:**
`ExamEase` is an AI-driven sentiment analysis tool designed specifically for recognizing student sentiments during exams. Utilizing the BERT model, it classifies student emotions into positive, negative, or neutral categories, providing a comprehensive understanding of student well-being.

## **Features:**

- **BERT Model Integration:** Advanced sentiment analysis using BERT.
- **Tri-sentiment Classification:** Recognizes positive, negative, and neutral sentiments.
- **User-Friendly Interface:** Easy-to-use command-line interface, and local hosting using Flask.
- **Focus on Well-being:** Aims to understand and cater to student emotional needs during exams.


## **Installation:**

1. Clone the repository:
  ```shell
 git clone https://github.com/yourusername/ExamEase.git
```

3. Navigate to the directory:
  ```shell
  cd ExamEase
```
3. Install requirements:
```shell
pip install -r requirements.txt
```

## **Usage:**
1. Train the model:
```
python3 train_bert.py
```
2. Run either in the terminal or on a local webpage
   1. Terminal:
      ```shell
      python3 main.py
      ```
   2. Local Host:
      ```shell
      python3 app.py
      ```
   open the webpage in your browser, generally its ```http://127.0.0.1:5000```


## **Contributions:**
Contributions are warmly welcomed! Feel free to create pull requests or open issues to discuss potential modifications or enhancements.

## **Disclaimer:**
ExamEase is primarily a research and educational tool, serving as no substitute for professional medical or psychological advice, diagnosis, or treatment. For any specific health-related concerns or uncertainties, consulting a qualified healthcare professional is strongly advised. The developers and contributors of ExamEase assume no liability for any inaccuracies, misinterpretations, or misuse of the tool.

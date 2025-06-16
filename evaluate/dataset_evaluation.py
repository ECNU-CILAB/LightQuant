from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai

def test_by_TextModel(dict):
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')

    features = tokenizer(['What is the stock price trend of Will Semiconductor today?',
                          'What is the stock price trend of Will Semiconductor today?'],
                         [dict['CSMD'], dict['CMIN-CN']],
                         padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        print(scores)

    score_1 = scores[0].item()
    score_2 = scores[1].item()

    return  score_1, score_2


def test_by_GPT(dict):

    # API KEY
    api_key = 'YOUR_OPENAI_API_KEY'

    # Comparison item
    dataset_1_description = dict['CSMD']
    dataset_2_description = dict['CMIN-CN']

    # Construct Prompt
    prompt = f"""
    You are an expert in data quality evaluation and natural language processing. 
    Given two datasets, evaluate their overall quality based on three criteria: coherence, information richness, and topic depth. 
    Provide a score from 0 to 100 for each dataset, where 0 indicates very poor quality and 100 indicates excellent quality. 
    Justify your scores briefly, highlighting strengths and weaknesses related to each criterion.

    Dataset 1: {dataset_1_description}
    Dataset 2: {dataset_2_description}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=10
    )

    scores_text = response['choices'][0]['message']['content']
    scores = scores_text.strip().split(',')


    score_1 = float(scores[0].strip())
    score_2 = float(scores[1].strip())

    print(f"Dataset 1 score: {score_1}")
    print(f"Dataset 2 score: {score_2}")

    return score_1, score_2

def main():

    # Here we have selected the data of the same stock (Weier Co., LTD.) on the same day (January 7, 2021) for comparison.
    # Of course, you can add more example here to compare different stocks and obtain a more comprehensive score, we just show you how to use this tool.
    dict = {'CSMD': "Weier Co., Ltd. conducted two block trades, with a total transaction value of 50.4861 million yuan. The transaction price for both was 240.41 yuan, representing a discount of 8.94% compared to today's closing price. Weir Co., Ltd. closed at 264.00 yuan today, up 4.32%. "
                          "The daily turnover rate was 2.20%, and the trading volume was 4.568 billion yuan. The net outflow of main force funds for the whole day was 410 million yuan. In the past five days, Weier Co., Ltd. has risen by 19.08% cumulatively."
                          " The total net inflow of funds in the past five days was 77.3472 million yuan, and the latest financing balance was 1.862 billion yuan, increasing by 344 million yuan in the past five days, with a growth rate of 22.64%.",
            'CMIN-CN': 'Semiconductor Core Daily | Fund Manager: New energy photovoltaic has at least ten times the potential, and the same goes for semiconductors. '
                          'Technology stocks that are expected to see a significant increase in performance. Review of northbound capital movements: Heavy selling of food and beverages, and pharmaceuticals. '
                          'Northbound funds have been selling off liquor stocks in large quantities but increasing their holdings in this sector (with stocks attached).'}

    TextModel_score1, TextModel_score2 = test_by_TextModel(dict)
    GPT_score1, GPT_score2 = test_by_GPT(dict)

    total_score1 = (TextModel_score1 - TextModel_score2) + GPT_score1
    total_score2 = (TextModel_score2 - TextModel_score1) + GPT_score2

if __name__ == '__main__':
    main()


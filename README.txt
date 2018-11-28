This program can be run using Powershell/Command Prompt or an IDE

Steps to run the python script:
1. Import/install the required libraries
    -nltk
    -sklearn
    -numpy
    -textblob

2. Run the script and input a name when prompted.

3. Ask any sushi related questions!

4. Exit the bot by saying any of the keywords: "end", "bye", "goodbye", "cya", "see ya", "see you", "farewell", "later", "quit"

Sample inputs:
>>Hi
>>What is sushi?
>>What are some types of sushi?
>>What is nigiri?
>>What is tuna?
>>What are sushi's origins?
>>I like <x>
>>I hate <x>
>>What do you know about me?
>>Who invented sushi?
>>How do you eat sushi?
>>Can you die from eating sushi?
>>Is it okay to eat fish raw?
>>uwu
>>Goodbye

Criteria:
User Data:  Bot remembers if you say something like "I like <x>" or "I hate <x>".
            Afterwards, if you ask,"do you remember what I like?" or "What do i hate?", 
            the bot will randomly choose one of the likes you told him about.

Knowledge Base: Bot incorporates knowledge through the use of tfidf and
                training. It does a database look-up and matches to most relative dialog.


GitHub:
https://github.com/xToyo/ZoroBot
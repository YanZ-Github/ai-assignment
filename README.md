# ai-assignment

## /reports
All generated JSON files as results with accuracy and detailed response for each question.

### /qwen-turbo-0624 and /qwen2-7b-instruct
Results for model 'qwen-turbo-0624' and 'qwen2-7b-instruct'
- aprompt: JSON files generated when using a prompt that works for improving the accuracy.
- bestprompt: JSON files generated when using the best prompt.
- noprompt: JSON files generated when prompt is not provided and system message (context and an assistant) is not refined; JSON files generated when prompt is not provided and system message (context and an assistant) is refined.

### /aprompt.txt
This .txt file stores the prompt that works for improving the accuracy.

### /bestprompt.txt
This .txt file stores the best prompt that works for improving the accuracy.

### /result.json
The .json file stores the runtime result.

### /sample.py
The fully-functional scripts
- Mainly use Alibaba model 'qwen2-7b-instruct' and 'qwen-turbo-0624'.
- System message is refined with more context and an assistant template example. 
- Implemented TODOs (calculate the accuracy of the result).

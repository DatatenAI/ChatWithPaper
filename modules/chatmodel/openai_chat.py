import optimize_openai

chat_paper_api = optimize_openai.ChatPaperAPI(
    model_name="gpt-3.5-turbo-16k",
    # model_name="gpt-3.5-turbo",
    top_p=1,
    temperature=0.0,
    apiTimeInterval=0.02)

chat_paper_api_short = optimize_openai.ChatPaperAPI(
    # model_name="gpt-3.5-turbo-16k",
    model_name="gpt-3.5-turbo",
    top_p=1,
    temperature=0.0,
    apiTimeInterval=0.02)
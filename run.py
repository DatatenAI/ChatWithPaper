from dotenv import load_dotenv

from chat_with_paper import handler

load_dotenv()

if __name__ == '__main__':

    handler('{"summary_id":"5e13cdbfc20453746daf8e99da766a74_中文"}')

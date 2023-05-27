import json

import flask
from dotenv import load_dotenv
from flask import Flask

from summary import handler

load_dotenv()
app = Flask(__name__)


@app.route('/invoke', methods=['get'])
def invoke():
    summary_id = flask.request.args.get('summary_id')
    if summary_id is None:
        return 'failed'
    handler(json.dumps({"summary_id": summary_id}))
    return 'success'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555)

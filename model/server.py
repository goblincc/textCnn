from tensorflow.keras import models
from flask import request, Flask
from tensorflow.keras.preprocessing import sequence
import config
from predict import sentence_index
import numpy as np
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
model = models.load_model('./modelfile')

@app.route('/', methods=['get','post'])
def post_Data():
    input_str = request.get_data(as_text=True)
    print(input_str)
    line_vector = sentence_index(input_str)
    result = 0
    if len(line_vector) >= 3:
        line_vector = sequence.pad_sequences([line_vector], maxlen=config.maxlen, padding='post', truncating='post')
        score = model.predict(line_vector)
        result = {
            "score": str(score[0][0])
        }
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    app.run(port=50001)
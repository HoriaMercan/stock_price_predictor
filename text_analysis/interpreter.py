try:
    from database.data import abbrevations
    from database.patterns import patterns
except:
    print("do preprocessing first")
    exit(1)
from config import config 
import spacy

class Interpreter:
    def __init__(self):
        self.tasks = []
        # Load the basic English model
        self.ner_model = spacy.load("en_core_web_sm")
        # On top of which we use gazeeters formed from our dataset
        ruler = self.ner_model.add_pipe("entity_ruler", config = config)
        ruler.add_patterns(patterns)
    def extract(self, request):
        # Empty the list of tasks
        self.tasks = []
        # Check if the request is give as a filepath
        if "\\" in request or "/" in request:
            filepath = request
            f = open(filepath, "r")
            # Create a single line request for easier processing
            request = ' '.join(f.readlines())
            f.close()
        # Set a standard lowercase setting
        request = request.lower()
        # Run the NER model on the request in order to extract
        # the organizations
        doc = self.ner_model(request)
        for entity in doc.ents:
            seq = entity.text
            lab = entity.label_
            print(seq + " --- " + lab)
        
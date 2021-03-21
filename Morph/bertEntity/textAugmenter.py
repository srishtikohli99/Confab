import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

def insert_augmentation(text):
    aug = naw.ContextualWordEmbsAug(
        model_path='./models/bert_base_uncased', action="insert")
    augmented_text = aug.augment(text)
    return augmented_text

def substitute_augmentation(text):
    aug = naw.ContextualWordEmbsAug(
        model_path='./models/bert_base_uncased', action="substitute")
    augmented_text = aug.augment(text)
    return augmented_text



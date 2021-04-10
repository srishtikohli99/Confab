entityExtractor=$(cat config.json | jq '.entityExtractor' | tr -d \")
entity=`cat config.json | jq '.entity'`
en="flair"
aug=`cat config.json | jq '.augment'`

if [ $aug = true ]
then
    echo "----------------------Augmenting data-----------------------------"
    python textAug/aug.py
fi

if [ $entity = true ]
then
    if [[ $entityExtractor == $en ]]
    then
        echo "-----------------Training Flair----------------------"
        python Morph/flairEntity/train.py
    fi
    echo "-----------------Training Bureau ML Classifier----------------------"
    python bureau/StatisticalClassifier.py
    echo "-----------------Training Bureau LSTM With Entities----------------------"
    python bureau/IntentWithEntity.py
fi
echo echo "-----------------Training Bureau LSTM Without Entites----------------------"
python bureau/IntentWithoutEntity.py

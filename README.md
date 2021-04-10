<p align="center"></p>
<p align="center">
    <img src="./images/poster.png" width="400">
    <h1 align="center">Confab</h1>
    <p align="center">Confab is a natural language understanding platform which helps laymen create and
optimize AI-powered interactive chatbots without worrying about the underlying architecture. Confab helps creating bots at scale using a plug and play methodology. Our tool beneﬁts those who are not backed by data science as well as data scientists who wish to deliver bots at scale
</p>
    <p align="center">
        <a></a>
        <br>
        <img src="https://img.shields.io/badge/status-WIP-yellow">
    </p>
</p>

## Architecture Diagram
<p align="center">
<img src="./images/trainingPipeline.png" width="800">
</p>

## Usage

### Training

### Testing

### REST Service

Once all the models are trained, Confab will start acting as bot-as-a-service.
REST endpoint is served using FastAPI.

```
uvicorn app:api --host 0.0.0.0 --port <PORT_NUMBER>  --reload
```



import numpy as np 
from river import metrics, stream, compose
import asyncio
import ESM_functions


def pool_embedding(embeddings, method='max'):
    pooled = np.zeros(len(embeddings[0]))
    for embedding in embeddings:
        if method=='max':
            pool = embedding.max(1)
        elif method=='avg':
            pool= embedding.mean(1)
        pooled = np.append(pooled, pool)
    return pooled[1:]


async def get_embed(ESM_model, batch_tokens, batch_strs):
    token_representations = ESM_model.embed_many(batch_tokens)
    token_reps = list(token_representations)
    embeddings = ESM_functions.pad_token_reps(token_reps, size=1000)
    embeddings = embeddings[1:]
    return pool_embedding(embeddings), batch_strs

def train_partial(stream, model):
    for data, target in stream:
        model = model.learn_one(data, target)

async def train_model(X, y, model):
    train_stream = stream.iter_array(
        X, y,
        feature_names = ['x{}'.format(j) for j in range(len(X))] 
    )
    return train_partial(train_stream, model)


async def sync_train_tasks(ESM_model, batch_tokens, batch_strs, X, y, model, metric):
    L = asyncio.gather(
        get_embed(ESM_model, batch_tokens, batch_strs),
        train_model(X, y, model, metric)
    )
    return L[0][0], L[0,1], L[1] 


async def run_train_model(data, ESM_model, classifier):
    model = compose.Pipeline(
        classifier
    )

    for batch_tokens, batch_strs in data.iter_train():
        X, y = asyncio.run(get_embed(ESM_model, batch_tokens, batch_strs))
        break
    i=0
    for batch_tokens, batch_strs in data.iter_train():
        next_X, next_y, model = await sync_train_tasks(ESM_model, batch_tokens, batch_strs, X, y, model)
        print('trained batch {}'.format(i))
        i+=1
        X = next_X
        y = next_y
    
    model = asyncio.run(train_model(X, y, model))
    print('trained batch {}'.format(i))
    return model

def test_partial(stream, model, metric):
    for data, target in stream:
        y_pred = model.predict_one(data)
        metric = metric.update(target, y_pred)
    return metric

async def test_model(X, y, model, metric):
    test_stream = stream.iter_array(
        X, y,
        feature_names = ['x{}'.format(j) for j in range(len(X))] 
    )
    return test_partial(test_stream, model, metric)

async def sync_test_tasks(ESM_model, batch_tokens, batch_strs, X, y, model, metric):
    L = asyncio.gather(
        get_embed(ESM_model, batch_tokens, batch_strs),
        test_model(X, y, model, metric)
    )
    return L[0][0], L[0,1], L[1] 

async def run_test_model(data, ESM_model, model, metric_type='accuracy'):
    if metric_type=='accuracy':
        metric = metrics.Accuracy()
    else:
        raise Exception('Invalid metric')

    for batch_tokens, batch_strs in data.iter_test():
        X, y = asyncio.run(get_embed(ESM_model, batch_tokens, batch_strs))
        break
    i=0
    for batch_tokens, batch_strs in data.iter_test():
        next_X, next_y, metric = await sync_test_tasks(ESM_model, batch_tokens, batch_strs, X, y, model, metric)
        print('tested batch {}'.format(i))
        i+=1
        X = next_X
        y = next_y
    
    metric = asyncio.run(test_model(X, y, model, metric))
    print('tested batch {}'.format(i))
    return metric.get()


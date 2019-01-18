registered와 casual 따로 cross_val_scorer구해서 각자의 hyperparameter 적용하였더니, 결과는 0.37136으로 0.001이상 안좋아졌다. validation결과가 완벽하게 test결과로 나타나지는 않는다

따라서 여기선 기존의 casual만 croass_val_scorer한 hypermeter를 동일하게 registered, casual 모델에 적용한다

확실히 둘을 구분하는 것이 성능은 좋게 나옴. 그렇지만 모델은 count 통합으로 validation했을 때 가장 이상적인 hypermeter적용
import json


class Tweet:
    def __init__(self, id, user_id, timestamp, location, text, friends, followers, favorites, score=0) -> None:
        super().__init__()
        self.score = score
        self.user_id = user_id
        self.followers = followers
        self.friends = friends
        self.text = text
        self.location = location
        self.timestamp = timestamp
        self.id = id
        self.favorites = favorites

    def features(self, feature_extractor, numeric=False):
        """
        :param numeric: If True; all features must be numeric
        :param feature_extractor: Converts tweet text to a vector with particular size
        :return: two list; (1) features, and (2) score
        """
        ret = []
        if numeric:
            ret.append(hash(self.id))
            ret.append(hash(self.user_id))
            ret.append(hash(self.timestamp))
            ret.append(hash(self.location))
        else:
            ret.append(self.id)
            ret.append(self.user_id)
            ret.append(self.timestamp)
            ret.append(self.location)

        ret.append(self.friends)
        ret.append(self.followers)
        ret.append(self.favorites)
        ret.extend(feature_extractor(self.text))
        return ret, self.score

    def __str__(self) -> str:
        return json.dumps(self.__dict__)

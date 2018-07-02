import json


class Tweet:
    def __init__(self, id, user_id, timestamp, location, text, friends, followers, favorites) -> None:
        super().__init__()
        self.user_id = user_id
        self.followers = followers
        self.friends = friends
        self.text = text
        self.location = location
        self.timestamp = timestamp
        self.id = id
        self.favorites = favorites

    def features(self):
        pass

    def __str__(self) -> str:
        return json.dumps(self.__dict__)

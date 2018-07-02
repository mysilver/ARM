import os
import xml.etree.ElementTree

from tweet import Tweet


def tab_splitor(line):
    return line.split('\t')


def read_paraphrased_tsv_files(directory, processor=None, reverse_columns=True, remove_html_tags=True):
    """
    Recursively reads the files in a directory and its sub-directories. 
    It is assumed that the first line starts with "# expression: " which indicates the expression which is paraphrased.
    :param reverse_columns: Reverse the order of columns in TSV files
    :param directory: The root directory where all files are located
    :param processor: 
    :return: dict
    """
    ret = {}
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            f = os.path.join(root, filename)
            with open(f, 'rt') as file:
                file_dataset = []
                lines = file.readlines()
                expression = lines[0].replace("# expression:", "")
                if remove_html_tags:
                    expression = expression.replace("</b>", "").replace("<b class='fixed'>", "")
                if processor:
                    expression = processor(expression)

                for i in range(1, len(lines)):
                    columns = tab_splitor(lines[i])
                    pc = []
                    for c in columns:
                        if processor:
                            pc.append(processor(c))
                        else:
                            pc.append(c)
                    if reverse_columns:
                        pc.reverse()
                    file_dataset.append(pc)

                if expression not in ret:
                    ret[expression] = []
                ret[expression].extend(file_dataset)

    return ret


def read_corpus(file, splitor=tab_splitor, processor=None):
    """
    Reads and loads all items in a text file,
    :param file: a text file
    :param splitor: a function to determine how to split each line of the file 
    :param processor: a function to preprocess each line of the code
    :return: list of processed lines
    """
    ret = []
    with open(file, 'rt') as f:
        for line in f.readlines():
            temp = []
            for item in splitor(line):
                if processor:
                    temp.append(processor(item))
                else:
                    temp.append(item)
            ret.append(temp)
    return ret


def tweet_xml_reader(file, dictionary=False):
    e = xml.etree.ElementTree.parse(file).getroot()

    if dictionary:
        ret = {}
    else:
        ret = []
    for atype in e.findall('Tweet'):
        id = atype.find('TweetID').text
        text = atype.find('TweetText').text
        time = atype.find('TweetTimeStamp').text
        location = atype.find('TweetLocation').text
        userid = atype.find('UserID').text
        friends = atype.find('UserFriendsCount').text
        followers = atype.find('UserFollowersCount').text
        favorites = atype.find('UserFavoritesCount').text
        tweet = Tweet(id=id, user_id=userid, timestamp=time, location=location, text=text, friends=friends,
                      followers=followers, favorites=favorites)
        if dictionary:
            ret[tweet.id] = tweet
        else:
            ret.append(tweet)

    return ret


for tweet in tweet_xml_reader("/media/may/Data/LinuxFiles/PycharmProjects/ARM/data/tweets-sample.xml", False):
    print(tweet)

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json
import pprint

credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials)

pp = pprint.PrettyPrinter(indent=4)

request_data = {'instances':
  [
    {
      'example_id': '1',
      'subreddit': 'news',
      'comment': 'I love reddit so much.'
    },
    {
      'example_id': '2',
      'subreddit': 'ireland',
      'comment': 'I love reddit so much.'
    },
    {
      'example_id': 'x3',
      'subreddit': 'pics',
      'comment': 'I love reddit so much.'
    },
  ]
}

print "/n...online predict test data.../n"
pp.pprint(request_data)

parent = 'projects/%s/models/%s/versions/%s' % ('pmc-analytical-data-mart', 'redditscore', 'v001')

response = api.projects().predict(body=request_data, name=parent).execute()

#print "response={0}".format(response)
print "/n...response.../n"
pp.pprint(response)

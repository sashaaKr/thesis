import time
from pprint import pprint
import utils.utils as thesisUtils
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'creds.json'

class GoogleSheetsClient:
  def __init__(self, spreadsheet_id, sheet_id, sheet_name):
    self.sheet_id = sheet_id
    self.sheet_name = sheet_name
    self.spreadsheet_id = spreadsheet_id

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    self.service = build('sheets', 'v4', credentials=creds)
    self.sheet =  self.service.spreadsheets()
    
  
  def write(self, values, range):
    self.sheet.values().update(
      spreadsheetId=self.spreadsheet_id,
      # range=f"{self.sheet_name}!A1", 
      range=f"{self.sheet_name}!{range}", 
      valueInputOption="USER_ENTERED", 
      body = { "values": values }
      # body={"values":[['hello', 'from', 'there']]}
      ).execute()

  # TODO: i can fetch cell len
  def colorize(self, indexes, text_len, row_index, col_index):
    sorted_indexes = sorted(indexes, key = lambda x: x[0])

    textFormatRuns = []

    for (start, end) in sorted_indexes:
      textFormatRuns.append({"startIndex":start,"format":{"foregroundColor":{"blue":1},"foregroundColorStyle":{"rgbColor":{"blue":1,}}}})
      if end == text_len: continue
      textFormatRuns.append({"startIndex":end})


    # pprint(textFormatRuns)
    requests = {
      'updateCells': {
        'rows': { 'values': [ { 'textFormatRuns': textFormatRuns } ] },
        'start': { 'sheetId': self.sheet_id, 'rowIndex': row_index, 'columnIndex': col_index },
        # 'range': { 'sheetId': self.sheet_id, 'startRowIndex': 0, 'endRowIndex': 1, 'startColumnIndex': 0, 'endColumnIndex': 1 },
        'fields': 'textFormatRuns(format)'
      }
    }
    body = { 'requests': [ requests ] }
    self.sheet.batchUpdate(
      spreadsheetId=self.spreadsheet_id,
      body=body
    ).execute()
  
  # def batch_colorize(self, data):
  #   values = []

  #   for indexes, text in data:
  #     text_len = len(text)
  #     textFormatRuns = []
  #     sorted_indexes = sorted(indexes, key = lambda x: x[0])

  #     for (start, end) in sorted_indexes:
  #       textFormatRuns.append({"startIndex":start,"format":{"foregroundColor":{"blue":1},"foregroundColorStyle":{"rgbColor":{"blue":1,}}}})
  #       if end == text_len: continue
  #       textFormatRuns.append({"startIndex":end})

  #     values.append({ 'textFormatRuns': textFormatRuns })
    
  #   requests = {
  #     'updateCells': {
  #       'rows': { 'values': values },
  #       # 'start': { 'sheetId': self.sheet_id, 'rowIndex': row_index, 'columnIndex': col_index },
  #       'range': { 'sheetId': self.sheet_id, 'startRowIndex': 0, 'endRowIndex': 217, 'startColumnIndex': 0, 'endColumnIndex': 1 },
  #       'fields': 'textFormatRuns(format)'
  #     }
  #   }
  #   body = { 'requests': [ requests ] }
  #   self.sheet.batchUpdate(
  #     spreadsheetId=self.spreadsheet_id,
  #     body=body
  #   ).execute()
  
  # def get_cell_text(self):



class BurchardResults:
  def __init__(
    self, 
    *, 

    burchard_corpus, 
    london_left_overs_corpus,
    zwickau_left_overs_corpus,

    burchard_vs_london_burchard_wrong_predictions, 
    burchard_vs_zwickau_burchard_wrong_predictions,

    burchard_vs_london_london_wrong_predictions,
    burchard_vs_zwickau_zwickau_wrong_predictions,

    london_predictions_by_burchard_vs_zwickau_classifier,
    zwickau_predictions_by_burchard_vs_london_classifier,
    ):
    self.sheet_id = '1AjK7zwT53CLma3P-uUy_zzHi2oGZvPinsOjWU7AGBfY'

    self.client = GoogleSheetsClient(self.sheet_id, 0, 'Burchard')
    self.london_left_overs_client = GoogleSheetsClient(self.sheet_id, 1931837305, 'London leftovers')
    self.zwickau_left_overs_client = GoogleSheetsClient(self.sheet_id, 482353935, 'Zwickau leftovers')

    self.burchard_corpus = burchard_corpus
    self.london_left_overs_corpus = london_left_overs_corpus
    self.zwickau_left_overs_corpus = zwickau_left_overs_corpus

    self.london_lef_overs_data = self.build_left_overs_data(self.london_left_overs_corpus)
    self.zwickau_left_overs_data = self.build_left_overs_data(self.zwickau_left_overs_corpus)
    
    self.burchard_vs_london_burchard_wrong_predictions = burchard_vs_london_burchard_wrong_predictions
    self.burchard_vs_zwickau_burchard_wrong_predictions = burchard_vs_zwickau_burchard_wrong_predictions

    self.burchard_vs_london_london_wrong_predictions = burchard_vs_london_london_wrong_predictions
    self.burchard_vs_zwickau_zwickau_wrong_predictions = burchard_vs_zwickau_zwickau_wrong_predictions

    self.london_predictions_by_burchard_vs_zwickau_classifier = london_predictions_by_burchard_vs_zwickau_classifier
    self.zwickau_predictions_by_burchard_vs_london_classifier = zwickau_predictions_by_burchard_vs_london_classifier
  
  def write(self):
    self.write_headers()
    self.write_paragraphs()
    self.colorize_burchard_shared_parts()
    self.write_vs_zwickau_predictions()
    self.write_vs_london_predictions()

    time.sleep(60)
    self.write_london_left_overs_paragraphs()
    self.colorize_london_leftovers_shared_parts()
    self.write_london_wrong_predictions()
    self.write_london_predictions_by_burchard_vs_zwickau_classifier()

    time.sleep(60)
    self.write_zwickau_left_overs_paragraphs()
    self.colorize_zwickau_leftovers_shared_parts()
    self.write_zwickau_wrong_predictions()
    self.write_zwickau_predictions_by_burchard_vs_london_classifier()
  
  def write_london_predictions_by_burchard_vs_zwickau_classifier(self):
    self.london_left_overs_client.write(self.london_predictions_by_burchard_vs_zwickau_classifier, 'D2')

  def write_zwickau_predictions_by_burchard_vs_london_classifier(self):
    self.zwickau_left_overs_client.write(self.zwickau_predictions_by_burchard_vs_london_classifier, 'D2')

  def write_london_wrong_predictions(self):
    predictions = [['London'] for i in range(157)]
    for wrong_prediction in self.burchard_vs_london_london_wrong_predictions:
      predictions[wrong_prediction.index][0] = 'Burchard'
    self.london_left_overs_client.write(predictions, 'C2')

  def write_zwickau_wrong_predictions(self):
    predictions = [['Zwickau'] for i in range(146)]
    for wrong_prediction in self.burchard_vs_zwickau_zwickau_wrong_predictions:
      predictions[wrong_prediction.index][0] = 'Burchard'
    self.zwickau_left_overs_client.write(predictions, 'C2')

  def build_left_overs_data(self, leftovers):
    return [ [ i, leftovers.map_to_original[i] ] for i in  leftovers.filter_short_p() ]

  def write_london_left_overs_paragraphs(self):
    self.london_left_overs_client.write(self.london_lef_overs_data, 'A2')

  def write_zwickau_left_overs_paragraphs(self):
    self.zwickau_left_overs_client.write(self.zwickau_left_overs_data, 'A2')

  def write_headers(self):
    self.client.write(
      [
        [
          'Burchard text (order by london)', 
          'London text', 
          'Zwickau text',
          'Burchard VS Zwickau classifier',
          'Burchard VS London classifier'
          ]
      ], 
      'A1'
      )

    self.london_left_overs_client.write(
      [
        [
          'Leftovers text',
          'Original Text',
          'Burchard VS London classification',
          'Burchard VS Zwickau classification'
        ]
      ],
      'A1'
      )
    self.zwickau_left_overs_client.write(
      [
        [
          'Letfovers text',
          'Original text',
          'Burchar VS Zwickau classification',
          'Burchard VS London classification'
        ]
      ],
      'A1'
    )

  def write_paragraphs(self):
    self.client.write(self.burchard_corpus.with_build_references(), 'A2')
  
  def write_vs_zwickau_predictions(self):
    predictions = [['Burchard'] for i in range(217)]
    for wrong_prediction in self.burchard_vs_zwickau_burchard_wrong_predictions:
      predictions[wrong_prediction.index][0] = 'Zwickau'

    self.client.write(predictions, 'D2')
  
  def write_vs_london_predictions(self):
    predictions = [['Burchard'] for i in range(217)]
    for wrong_prediction in self.burchard_vs_london_burchard_wrong_predictions:
      predictions[wrong_prediction.index][0] = 'London'

    self.client.write(predictions, 'E2')

  def colorize_zwickau_leftovers_shared_parts(self):
    first_itter = True
    for index, d in enumerate(self.zwickau_left_overs_data):
      if not first_itter and index % 25 == 0: 
        print('sleeping')
        time.sleep(60)
      first_itter = False

      shared = thesisUtils.get_indexes_of_shard_word(d[0], d[1])
      p_for1 = []
      p_for2 = []
      for s in shared:
        p_for1.append(s[1])
        p_for2.append(s[2])
      self.zwickau_left_overs_client.colorize(p_for1, len(d[0]), 1 + index, 0)
      self.zwickau_left_overs_client.colorize(p_for2, len(d[1]), 1 + index, 1)

  def colorize_london_leftovers_shared_parts(self):
    first_itter = True
    for index, d in enumerate(self.london_lef_overs_data):
      if not first_itter and index % 25 == 0: 
        print('sleeping')
        time.sleep(60)
      first_itter = False

      shared = thesisUtils.get_indexes_of_shard_word(d[0], d[1])
      p_for1 = []
      p_for2 = []
      for s in shared:
        p_for1.append(s[1])
        p_for2.append(s[2])
    
      self.london_left_overs_client.colorize(p_for1, len(d[0]), 1 + index, 0)
      self.london_left_overs_client.colorize(p_for2, len(d[1]), 1 + index, 1)

  def colorize_burchard_shared_parts(self):
    first_itter = True
    for index, match in enumerate(self.burchard_corpus.matches_used_for_build):
      if not first_itter and index % 25 == 0: 
        print('sleeping')
        time.sleep(60)
      first_itter = False
        
      shared = thesisUtils.get_indexes_of_shard_word(match.original_text, match.match_text)
      p_for1 = []
      p_for2 = []
      for s in shared:
        p_for1.append(s[1])
        p_for2.append(s[2])
    
      self.client.colorize(p_for1, len(match.original_text), 1 + index, 1)
      self.client.colorize(p_for2, len(match.match_text), 1 + index, 2)
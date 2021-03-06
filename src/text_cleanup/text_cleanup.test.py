import unittest
import text_cleanup.text_cleanup as text_cleanup

original_vs_expexted = [
    [
        "(112vb) Incipit descripcio terre sancte", # original text
        "incipit descriptio terre sancte" # expected text after cleanup
    ],
    [
        "Bericus Ultra Sydonem ad 4 leucas est (117r) nobilis illa et antiqua civitas Beritus, ubi dicitur etiam dominus predicasse, ubi Iudei, facta ymagine salvatoris de pasta, ei illuserunt et tandem crucifigentes ab ea sanguine in magna quantitate extraxerunt, qui usque hodie in multis locis venerationi habetur",
        "bericus ultra sidonem ad leucas est nobilis ila et antiqua ciuitas beritus ubi dicitur etiam dominus predicasse ubi iudei facta imagine saluatoris de pasta ei iluserunt et tandem crucifigentes ab ea sanguine in magna quantitate extraxerunt qui usque hodie in multis locis uenerationi habetur"
    ],
    [
        "In(123r)de ad ½ leucam est Dothaim, de qua dictum est supra.",
        "inde ad leucam est dotaim de qua dictum est supra"
    ],
    [
        # "  Qui mons per totam fere Galileam videtur, pulcher valde et firmus.",
        'Mons Bethulie Inde ad 3 leucis est mons Bethulie, ubi Iudit occidit Holofernem.  Qui mons per totam fere Galileam videtur, pulcher valde et firmus.',
        "mons betulie inde ad leucis est mons betulie ubi iudit occidit holofernem qui mons per totam fere galileam uidetur pulcer ualde et firmus"
    ],
    [
        "Iesrahel pulchrum habet prospectum per totam Galileam usque ad Carmelum et montes Phenicie montemque Thabor montemque Galaat ultra Iordanem et totum montem Effraym iterum usque ad Carmelum.",
        "iesrahel pulcrum habet prospectum per totam galileam usque ad carmelum et montes penicie montemque tabor montemque galaat ultra iordanem et totum montem efraim iterum usque ad carmelum"
    ],
    [
        "De Balonia recedens ductus sum ad locum ubi erant 6 leones et 6 elephantes et 60 strutones in curia et onagri plures.",
        "de balonia recedens ductus sum ad locum ubi erant leones et elepantes et strutones in curia et onagri plures"
    ],
    [
        # removes double ll
        'Ego autem credo illud pocius esse Chanaam filii Cham filii Noe vel alicuius filiorum eius, qui ad terram circa loca illa habitasse.',
        'ego autem credo ilud pocius esse canaam filii cam filii noe aut alicuius filiorum eius qui ad terram circa loca ila habitasse'
    ],
    [
        # removes double mm
        'Fures qui minora furta committunt, ut non sint digni suspendio, vel malefactores alii commitentes crimina minora, castrantur, ne filios generent, qui patrum crimina imitentur et hec videtur mihi una causa, quare tot meretrices sunt in partibus illis.',
        'fures qui minora furta comitunt ut non sint digni suspendio aut malefactores alii comitentes crimina minora castrantur ne filios generent qui patrum crimina imitentur et hec uidetur mii una causa quare tot meretrices sunt in partibus ilis'
    ],
    [
        # removes double tt
        'Via occidentis Hec autem linea ad litteram est via illa, de qua dicitur Thobias 1.',
        'uia occidentis hec autem linea ad literam est uia ila de qua dicitur tobias'
    ],
    [
        #change z->s
        'Accedit insuper ad easdem piscinas fons Gyon inferior, situs in pede montis Gyon contra agrum Acheldemach iuxta locum, ubi stetit Rapsaces, quando domino exprobavit et Ezechieli et sedenti populo super murum.',
        'accedit insuper ad easdem piscinas fons gion inferior situs in pede montis gion contra agrum aceldemac iuxta locum ubi stetit rapsaces quando domino exprobauit et esecieli et sedenti populo super murum'
    ],
    [
        # change ih -> i
        'Sarraceni Machmetum predicant et legem eius servant, dominum Ihesum Christum maximum prophetarum dicunt et eundem de spiritu sancto conceptum et de virgine natum, sed neFehler!',
        'sarraceni macmetum predicant et legem eius seruant dominum iesum cristum maximum propetarum dicunt et eundem de spiritu sancto conceptum et de uirgine natum sed nefehler'
    ],
    [
        'Secunda Palestina est cuius metropolis est Cesarea Palestine sive Cesarea Martania cum tota terra <palasti> Philistim, incipiens a predicta Petra Incisa sive Castro Peregrinorum et extendens se usque Gazam et Gerazam.',
        'secunda palestina est cuius metropolis est cesarea palestine siue cesarea martania cum tota terra palasti pilistim incipiens a predicta petra incisa siue castro peregrinorum et extendens se usque gasam et gerasam'
    ],
    [
        # change ae -> e
        'Inde 4 leucis est civitas antiqua Asor dicta in qua habitavit rex Iabin ille potens, qui cum 24 regibus pugnavit contra Iosue et Israel ad Aquas Maron.',
        'inde leucis est ciuitas antiqua asor dicta in qua habitauit rex iabin ile potens qui cum regibus pugnauit contra iosue et isrel ad aquas maron'
    ],
    [
        # change ch -> c
        'Inde per Anthiochiam veni ad montana magna que sunt promuctorium Sicilie quod dicitur Ratzgenerz i[d est] caput porci, et sic applicuimus in portu Palis et venimus ad Ayaz.',
        'inde per antiociam ueni ad montana magna que sunt promuctorium sicilie quod dicitur ratsgeners id est caput porci et sic applicuimus in portu palis et uenimus ad aias'
    ],
    [
        # atque -> et
        'Alius mons Seyr coniungitur cum deserto Pharan, quem longo tempore circuierunt et de illo dicitur Deuteronomius 3o: „In Monte Seyr olim habitaverunt horrei, quibus eiectis atque deletis habitaverunt filii Esau pro eis“.',
        'alius mons seir coniungitur cum deserto paran quem longo tempore circuierunt et de ilo dicitur deuteronomius in monte seir olim habitauerunt horrei quibus eiectis et deletis habitauerunt filii esau pro eis'
    ],
    [
        # change vel -> aut
        'Quem enim hora diei vel noctis per totum anni circulum in qua non recolit cantando legendo psallendo predicando et meditando omnis devotus Christianus que facta sunt vel scripta in hac terra et civitatibus et locis eius.',
        'quem enim hora diei aut noctis per totum anni circulum in qua non recolit cantando legendo psalendo predicando et meditando omnis deuotus cristianus que facta sunt aut scripta in hac terra et ciuitatibus et locis eius'
    ],
    [
        # remove number with o postfix
        "Alia insuper vice scilicet anno domini 1283o in festo Omnium Sanctorum sub divo dormiens in eodem monte cum aliis multis fui cum ipsis rore penitus infusus nocte illa.",
        "alia insuper uice scilicet anno domini in festo omnium sanctorum sub diuo dormiens in eodem monte cum aliis multis fui cum ipsis rore penitus infusus nocte ila"
    ],
    [
        # change th -> t
        'Thiberias De Bethulia ad 5 leucas et plus supra mare Galilee est Tyberias civitas, a qua idem mare Thyberiadis dicitur, olim Cenereth dicta, a qua etiam mare Cenereth dictum.',
        'tiberias de betulia ad leucas et plus supra mare galilee est tiberias ciuitas a qua idem mare tiberiadis dicitur olim ceneret dicta a qua etiam mare ceneret dictum'
    ],
    [
        # change y -> i
        "Fuit autem in campo per longum disposita tendens ab austro in aquilonem sub monte Antylibano inter ipsum et mare spaciosa valde.",
        "fuit autem in campo per longum disposita tendens ab austro in aquilonem sub monte antilibano inter ipsum et mare spatiosa ualde"
    ],
    [
        # remove vii
        "Excrescit autem mare illud interdum de resolutione nivium de Lybano et aliis montanis, de inundatione Iordanis et torrentis Iaboth et Armon et Zareth et de pluviis cadentibus in Galilea, monte Galaad, terra Moab et Amon et Seyr de quibus omnibus aque descendunt per Jordanem ad mare istud reliquos autem vii viii vi",
        "excrescit autem mare ilud interdum de resolutione niuium de libano et aliis montanis de inundatione iordanis et torrentis iabot et armon et saret et de pluuiis cadentibus in galilea monte galaad terra moab et amon et seir de quibus omnibus aque descendunt per iordanem ad mare istud reliquos autem"
    ],
    [
        # should remove iii, ii, iiii
        'De iherusalem ad iii leucas vel plus contra occidentem est emaus ubi dominus iiii agnitus est in fractione ii panis.',
        'de ierusalem ad leucas aut plus contra occidentem est emaus ubi dominus agnitus est in fractione panis'
    ],
    [
        # it should remove single v
        'Est autem in rei veritate campus iste planicies galilee que cingitur ab oriente mari galilee et yordane ab austro monte effraym et samaria, ab occidente partim monte effraym, partim monte Carmelo, ab aquilone montibus phenicie qui pheniciam dividunt et galileam, et habet in longitudine plene v miliaria tentonica vel amplius, fertilis supra modum vini et frumenti et omnium bonorum que gignere solet humus, et aere saluberrimo, ita quod videtur michi quod terram non viderim meliorem, si esset in manu Christianorum.',
        'est autem in rei ueritate campus iste planicies galilee que cingitur ab oriente mari galilee et iordane ab austro monte efraim et samaria ab occidente partim monte efraim partim monte carmelo ab aquilone montibus penicie qui penitiam diuidunt et galileam et habet in longitudine plene miliaria tentonica aut amplius fertilis supra modum uini et frumenti et omnium bonorum que gignere solet humus et ere saluberrimo ita quod uidetur mici quod terram non uiderim meliorem si esset in manu cristianorum'
    ],
    [
        # it should remove some english mark
        'De isto fonte ad dimidiam leucam yezrahel civitas quondam regalis nunc vix habet xx domos (236v is empty, 237r) in pede occidentali montis gelboe sita ubi adhuc vidi agrum naboth yezrahelite.',
        'de isto fonte ad dimidiam leucam iesrahel ciuitas quondam regalis nunc uix habet xx domos in pede occidentali montis gelboe sita ubi adhuc uidi agrum nabot iesrahelite'
    ],
    [
        # it should remove some english mark
        'Videtur enim velle quod murus civitatis et superficies montis erant equales et desuper civitas edificata (237v is empty, 238r) quod non erat. murus enim civitatis erat fere in pede montis firmus valde turribus et interius mons paulatim in altum surgens et circumseptus edificiis quasi botrus vue granis consitus et tendens in.',
        'uidetur enim aute quod murus ciuitatis et superficies montis erant equales et desuper ciuitas edificata quod non erat murus enim ciuitatis erat fere in pede montis firmus ualde turribus et interius mons paulatim in altum surgens et circumseptus edificiis quasi botrus uue granis consitus et tendens in'
    ],
    [
        # it should remove <q>
        'De monte Zacharie 2 leucis contra orientem fere via, que ducit Thecuam, sanctus Karychon abbas sepultus est <q> cum suis monachis, qui omnes eo migrante migraverunt et ostenditur ibidem usque hodie sepultura de Bethleem 2bus leucis.',
        'de monte sacarie leucis contra orientem fere uia que ducit tecuam sanctus karicon abbas sepultus est cum suis monacis qui omnes eo migrante migrauerunt et ostenditur ibidem usque hodie sepultura de betleem bus leucis'
    ],
    [
        # it should remove ix
        'De isto opido usque ad cumulos yordanis contra orientem sunt fere ix leuce',
        'de isto opido usque ad cumulos iordanis contra orientem sunt fere leuce'
    ],
    [
        # it should remove [n]
        'Est [n] insuper montana que sunt circa campestria iericho sunt alta supra modum et palatim descendant usque ad montem oliveti.',
        'est insuper montana que sunt circa campestria ierico sunt alta supra modum et palatim descendant usque ad montem oliueti'
    ],
    [
        # it should remove <n>
        'Comedi tamen <n> in ea cum monachis missa dicta ad sanctum Bartholomeum.',
        'comedi tamen in ea cum monacis missa dicta ad sanctum bartolomeum'
    ]
]

original_vs_expected_for_without_jvtext_and_word_replacement = [
  [
    # remove vii
    "Excrescit autem mare illud interdum de resolutione nivium de Lybano et aliis montanis, de inundatione Iordanis et torrentis Iaboth et Armon et Zareth et de pluviis cadentibus in Galilea, monte Galaad, terra Moab et Amon et Seyr de quibus omnibus aque descendunt per Jordanem ad mare istud reliquos autem vii viii vi",
    "Excrescit autem mare illud interdum de resolutione nivium de Lybano et aliis montanis de inundatione Iordanis et torrentis Iaboth et Armon et Zareth et de pluviis cadentibus in Galilea monte Galaad terra Moab et Amon et Seyr de quibus omnibus aque descendunt per Jordanem ad mare istud reliquos autem"
  ]
]
class TestTextCleanupMethods(unittest.TestCase):

  def test_cleanup(self):
    self.maxDiff = None
    for original, expected in original_vs_expexted:
      result = text_cleanup.create_corpus_by_line(text_cleanup.jvtext(original))
      self.assertEqual(result, [expected])

  def test_cleanup_without_jvtext_and_word_replacement(self):  
    self.maxDiff = None
    for original, expected in original_vs_expected_for_without_jvtext_and_word_replacement:
      result = text_cleanup.cleanup(original, False)
      self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
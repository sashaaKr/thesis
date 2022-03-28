import unittest
import text_cleanup.text_cleanup as text_cleanup

original_vs_expexted = [
    # [
    #     "(112vb) Incipit descripcio terre sancte",
    #     "Incipit descriptio terre sancte"
    # ],
    # [
    #     "Bericus Ultra Sydonem ad 4 leucas est (117r) nobilis illa et antiqua civitas Beritus, ubi dicitur etiam dominus predicasse, ubi Iudei, facta ymagine salvatoris de pasta, ei illuserunt et tandem crucifigentes ab ea sanguine in magna quantitate extraxerunt, qui usque hodie in multis locis venerationi habetur",
    #     "Bericus Ultra Sydonem ad leucas est nobilis ila et antiqua civitas Beritus ubi dicitur etiam dominus predicasse ubi Iudei facta ymagine salvatoris de pasta ei iluserunt et tandem crucifigentes ab ea sanguine in magna quantitate extraxerunt qui usque hodie in multis locis venerationi habetur"
    # ],
    # [
    #     "In(123r)de ad ½ leucam est Dothaim, de qua dictum est supra.",
    #     "Inde ad leucam est Dothaim de qua dictum est supra"
    # ],
    # [
    #     "  Qui mons per totam fere Galileam videtur, pulcher valde et firmus.",
    #     "Qui mons per totam fere Galileam videtur pulcher valde et firmus"
    # ],
    # [
    #     "Iesrahel pulchrum habet prospectum per totam Galileam usque ad Carmelum et montes Phenicie montemque Thabor montemque Galaat ultra Iordanem et totum montem Effraym iterum usque ad Carmelum.",
    #     "Iesrahel pulchrum habet prospectum per totam Galileam usque ad Carmelum et montes penicie montemque Thabor montemque Galaat ultra Iordanem et totum montem Efraym iterum usque ad Carmelum"
    # ],
    # [
    #     "De Balonia recedens ductus sum ad locum ubi erant 6 leones et 6 elephantes et 60 strutones in curia et onagri plures.",
    #     "De Balonia recedens ductus sum ad locum ubi erant leones et elepantes et strutones in curia et onagri plures"
    # ],
    # [
    #     # removes double ll
    #     'Ego autem credo illud pocius esse Chanaam filii Cham filii Noe vel alicuius filiorum eius, qui ad terram circa loca illa habitasse.',
    #     'Ego autem credo ilud pocius esse Chanaam filii Cham filii Noe vel alicuius filiorum eius qui ad terram circa loca ila habitasse'
    # ],
    # [
    #     # removes double mm
    #     'Fures qui minora furta committunt, ut non sint digni suspendio, vel malefactores alii commitentes crimina minora, castrantur, ne filios generent, qui patrum crimina imitentur et hec videtur mihi una causa, quare tot meretrices sunt in partibus illis.',
    #     'Fures qui minora furta comitunt ut non sint digni suspendio vel malefactores alii comitentes crimina minora castrantur ne filios generent qui patrum crimina imitentur et hec videtur mii una causa quare tot meretrices sunt in partibus ilis'
    # ],
    # [
    #     # removes double tt
    #     'Via occidentis Hec autem linea ad litteram est via illa, de qua dicitur Thobias 1.',
    #     'Via occidentis Hec autem linea ad literam est via ila de qua dicitur Thobias'
    # ],
    # [
    #     #change z->s
    #     'Accedit insuper ad easdem piscinas fons Gyon inferior, situs in pede montis Gyon contra agrum Acheldemach iuxta locum, ubi stetit Rapsaces, quando domino exprobavit et Ezechieli et sedenti populo super murum.',
    #     'Accedit insuper ad easdem piscinas fons Gyon inferior situs in pede montis Gyon contra agrum Acheldemach iuxta locum ubi stetit Rapsaces quando domino exprobavit et Esechieli et sedenti populo super murum'
    # ],
    # [
    #     # change ih -> i
    #     'Sarraceni Machmetum predicant et legem eius servant, dominum Ihesum Christum maximum prophetarum dicunt et eundem de spiritu sancto conceptum et de virgine natum, sed neFehler!',
    #     'Sarraceni Machmetum predicant et legem eius servant dominum iesum Christum maximum propetarum dicunt et eundem de spiritu sancto conceptum et de virgine natum sed neFehler'
    # ],
    # [
    #     'Secunda Palestina est cuius metropolis est Cesarea Palestine sive Cesarea Martania cum tota terra <palasti> Philistim, incipiens a predicta Petra Incisa sive Castro Peregrinorum et extendens se usque Gazam et Gerazam.',
    #     'Secunda Palestina est cuius metropolis est Cesarea Palestine sive Cesarea Martania cum tota terra palasti pilistim incipiens a predicta Petra Incisa sive Castro Peregrinorum et extendens se usque Gasam et Gerasam'
    # ],
    # [
    #     # change ae -> e
    #     'Inde 4 leucis est civitas antiqua Asor dicta in qua habitavit rex Iabin ille potens, qui cum 24 regibus pugnavit contra Iosue et Israel ad Aquas Maron.',
    #     'Inde leucis est civitas antiqua Asor dicta in qua habitavit rex Iabin ile potens qui cum regibus pugnavit contra Iosue et Isrel ad Aquas Maron'
    # ],
    # [
    #     # change ch -> c
    #     'Inde per Anthiochiam veni ad montana magna que sunt promuctorium Sicilie quod dicitur Ratzgenerz i[d est] caput porci, et sic applicuimus in portu Palis et venimus ad Ayaz.',
    #     'Inde per Anthiociam veni ad montana magna que sunt promuctorium Sicilie quod dicitur Ratsgeners id est caput porci et sic applicuimus in portu Palis et venimus ad Ayas'
    # ]
    [
        # atque -> et
        'Alius mons Seyr coniungitur cum deserto Pharan, quem longo tempore circuierunt et de illo dicitur Deuteronomius 3o: „In Monte Seyr olim habitaverunt horrei, quibus eiectis atque deletis habitaverunt filii Esau pro eis“.',
        'Alius mons Seyr coniungitur cum deserto paran quem longo tempore circuierunt et de ilo dicitur Deuteronomius o In Monte Seyr olim habitaverunt horrei quibus eiectis et deletis habitaverunt filii Esau pro eis'
    ]
]

class TestTextCleanupMethods(unittest.TestCase):

    def test_cleanup(self):
        for original, expected in original_vs_expexted:
            result = text_cleanup.cleanup(original)
            # print(result)
            self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
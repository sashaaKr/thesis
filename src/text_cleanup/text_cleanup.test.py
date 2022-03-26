import unittest
import text_cleanup.text_cleanup as text_cleanup

original_vs_expexted = [
    [
        "(112vb) Incipit descripcio terre sancte",
        "Incipit descriptio terre sancte"
    ],
    [
        "Bericus Ultra Sydonem ad 4 leucas est (117r) nobilis illa et antiqua civitas Beritus, ubi dicitur etiam dominus predicasse, ubi Iudei, facta ymagine salvatoris de pasta, ei illuserunt et tandem crucifigentes ab ea sanguine in magna quantitate extraxerunt, qui usque hodie in multis locis venerationi habetur",
        "Bericus Ultra Sydonem ad leucas est nobilis illa et antiqua civitas Beritus ubi dicitur etiam dominus predicasse ubi Iudei facta ymagine salvatoris de pasta ei illuserunt et tandem crucifigentes ab ea sanguine in magna quantitate extraxerunt qui usque hodie in multis locis venerationi habetur"
    ],
    [
        "In(123r)de ad ½ leucam est Dothaim, de qua dictum est supra.",
        "Inde ad leucam est Dothaim de qua dictum est supra"
    ],
    [
        "  Qui mons per totam fere Galileam videtur, pulcher valde et firmus.",
        "Qui mons per totam fere Galileam videtur pulcher valde et firmus"
    ],
    [
        "Iesrahel pulchrum habet prospectum per totam Galileam usque ad Carmelum et montes Phenicie montemque Thabor montemque Galaat ultra Iordanem et totum montem Effraym iterum usque ad Carmelum.",
        "Iesrahel pulchrum habet prospectum per totam Galileam usque ad Carmelum et montes Phenicie montemque Thabor montemque Galaat ultra Iordanem et totum montem Efraym iterum usque ad Carmelum"
    ],
    [
        "De Balonia recedens ductus sum ad locum ubi erant 6 leones et 6 elephantes et 60 strutones in curia et onagri plures.",
        "De Balonia recedens ductus sum ad locum ubi erant leones et elephantes et strutones in curia et onagri plures"
    ]
]

class TestTextCleanupMethods(unittest.TestCase):

    def test_cleanup(self):
        for original, expected in original_vs_expexted:
            self.assertEqual(text_cleanup.cleanup(original), expected)

if __name__ == '__main__':
    unittest.main()
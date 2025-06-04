namespace DecisionTree_1.Controllers;

using DecisionTree.Model.DataSet;
using DecisionTree.Model.Helper;
using DecisionTree.Model.Model.StabloKlasificatori.Helper;
using Microsoft.AspNetCore.Mvc;
using static StabloKlasifikator2;

[ApiController]
[Route("api/[controller]/[action]")]
public class DecisionTreeController : ControllerBase
{
    private readonly ExcelHelper _ucitavac;

    public DecisionTreeController(ExcelHelper ucitavac)
    {
        _ucitavac = ucitavac;
    }

    [HttpGet]
    public IActionResult Play()
    {
        return Pokreni(new()
        {
            PutanjaDoFajla = "Files/play1.xlsx",
            CiljnaVarijabla = "Play",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new (){
                MaxDepth = 5,
                MinSamples = 5
            }
        });
    }

    [HttpGet]
    public IActionResult Cardio()
    {
        return Pokreni(new()
        {
            PutanjaDoFajla = "Files/cardio1.xlsx",
            CiljnaVarijabla = "dcRyth",
            TestProcenat = 0.3,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 5,
                MinSamples = 5
            }
        });
    }

    [HttpGet]
    public IActionResult Sales()
    {
        StabloZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/Sales3.xlsx",
            CiljnaVarijabla = "SalesCategory",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 5,
                MinSamples = 5,
                BrojGrupaZaNumericke = 5
            }
        };

        MojDataSet fullDataSet0 = _ucitavac.Ucitaj(zahtjev.PutanjaDoFajla);

        MojDataSet fullDataSet = fullDataSet0.Clone();

        var (q1, q3) = KvartilaHelper.IzracunajKvartile(fullDataSet.Podaci, "OutletSales");

        fullDataSet.DodajKolonuKategorijski("SalesCategory", red =>
        {
            if (!red.Atributi.TryGetValue("OutletSales", out var attr) || !attr.Broj.HasValue)
                return null;

            var val = attr.Broj.Value;
            if (val < q1) return "Low";
            if (val > q3) return "High";
            return "Medium";
        });

        fullDataSet.CiljnaKolona = "SalesCategory";
        fullDataSet.IskljuciAtribute("OutletSales");

        fullDataSet.TransformNumerickuKolonuPoGrupi(
            nazivKolone: "Weight",
            grupnaKolona1: "ProductType",
            grupnaKolona2: "OutletType",
            transformacija: (stara, vrijednostiGrupe) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiGrupe),
            opisTransformacijeZaHistoriju: "medijana"
        );

        fullDataSet.TransformirajKolonuNumericku("Weight", (stara, vrijednostiKolone) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiKolone));


        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.PodijeliStratifikovano(zahtjev.TestProcenat, random_state: 42);

        StabloKlasifikator2 stablo = new StabloKlasifikator2(treningSet, zahtjev.KlasifikatorParamteri);
        GraphvizVisualizerHelper.MakeDotFile(stablo.korijen, "Files/Sales3");

        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
        });
    }


    [HttpGet]
    public IActionResult SalesMultiple()
    {
        var fajl = "Files/Sales3.xlsx";
        var ciljnaKolona = "SalesCategory";
        var testProcenat = 0.2;

        MojDataSet fullDataSet = _ucitavac.Ucitaj(fajl);

        var (q1, q3) = KvartilaHelper.IzracunajKvartile(fullDataSet.Podaci, "OutletSales");

        fullDataSet.DodajKolonuKategorijski(ciljnaKolona, red =>
        {
            if (!red.Atributi.TryGetValue("OutletSales", out var attr) || !attr.Broj.HasValue)
                return null;

            var val = attr.Broj.Value;
            if (val < q1) return "Low";
            if (val > q3) return "High";
            return "Medium";
        });

        fullDataSet.SetCiljnaVarijabla(ciljnaKolona);
        fullDataSet.IskljuciAtribute("OutletSales");

        fullDataSet.TransformNumerickuKolonuPoGrupi(
            nazivKolone: "Weight",
            grupnaKolona1: "ProductType",
            grupnaKolona2: "OutletType",
            transformacija: (stara, grupe) => stara ?? MedianHelper.IzracunajMedijan(grupe),
            opisTransformacijeZaHistoriju: "medijana"
        );

        fullDataSet.TransformirajKolonuNumericku("Weight", (stara, kolona) => stara ?? MedianHelper.IzracunajMedijan(kolona));

        // za stablo ne treba one-hot encoding
        // fullDataSet.NapraviOneHotEncodingSveKolone();

        // Petlja kroz više konfiguracija stabla
        var rezultati = new List<object>();

        for (int maxDepth = 3; maxDepth <= 10; maxDepth++)
        {
            var parametri = new StabloKlasifikatorParamteri
            {
                MaxDepth = maxDepth,
                MinSamples = 5
            };

            var (treningSet, testSet) = fullDataSet.Podijeli(testProcenat, random_state: 42);
            var stablo = new StabloKlasifikator2(treningSet, parametri);
            var rezultat = fullDataSet.Evaluiraj(stablo, testSet);

            rezultati.Add(new
            {
                parametri.MaxDepth,
                rezultat.Accuracy,
                AvgF1 = rezultat.AvgF1Score,
                rezultat.VrijemeTreniranjaSek,
                rezultat.VrijemeEvaluacijeSek
            });
        }

        return Ok(rezultati);
    }

    [HttpGet]
    public IActionResult Mushroom()
    {
        return Pokreni(new()
        {
            PutanjaDoFajla = "Files/mushroom1.xlsx",
            CiljnaVarijabla = "class",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 6,
                MinSamples = 6
            }
        });
    }

    [HttpGet]
    public IActionResult AdultCensusIncome()
    {
        return Pokreni(new()
        {
            PutanjaDoFajla = "Files/adult-census-income.xlsx",
            CiljnaVarijabla = "class",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 8,
                MinSamples = 4
            }
        });
    }
    [HttpGet]
    public IActionResult Pokreni(
        [FromQuery] StabloZahtjev zahtjev)
    {
        MojDataSet fullDataSet = _ucitavac.Ucitaj(zahtjev.PutanjaDoFajla, zahtjev.CiljnaVarijabla);
        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        StabloKlasifikator2 stablo = new StabloKlasifikator2(treningSet, zahtjev.KlasifikatorParamteri);
        GraphvizVisualizerHelper.MakeDotFile(stablo.korijen, zahtjev.PutanjaDoFajla);
        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
        });
    }

    public class StabloZahtjev
    {
        public string PutanjaDoFajla { get; set; } = string.Empty;

        public string CiljnaVarijabla { get; set; } = string.Empty;

        public double TestProcenat { get; set; } = 0.2;

        public StabloKlasifikatorParamteri KlasifikatorParamteri { get; set; } = new();
    }
}

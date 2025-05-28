using DecisionTree.Model.DataSet;
using DecisionTree.Model.Helper;
using DecisionTree.Model.Model;

public class CvorStabla
{
    /// <summary>
    /// Atribut po kojem se vrši grananje u ovom čvoru (null ako je list).
    /// </summary>
    public required string? Atribut { get; set; }

    /// <summary>
    /// Djeca ovog čvora, indeksirana po vrijednosti atributa.
    /// </summary>
    public Dictionary<string, CvorStabla> Djeca { get; set; } = new();

    /// <summary>
    /// Vrijednost ciljne klase ako je čvor list (null ako nije list).
    /// </summary>
    public required string? Klasa { get; set; }

    /// <summary>
    /// Granična vrijednost (za numeričke atribute).
    /// </summary>
    public required double? Threshold { get; set; }

    /// <summary>
    /// Tip atributa (kategorijski ili numerički).
    /// </summary>
    public required TipAtributa? Tip { get; set; }

    /// <summary>
    /// Da li je atribut numerički.
    /// </summary>
    public bool IsNumericki => Tip == TipAtributa.Numericki;

    /// <summary>
    /// Da li je čvor list (nema djece).
    /// </summary>
    public bool JeList => Klasa != null;

    /// <summary>
    /// Prazan konstruktor za ručno kreiranje.
    /// </summary>
    private CvorStabla() { }

    /// <summary>
    /// Kreira list čvor sa zadatom klasom.
    /// </summary>
    public static CvorStabla NapraviList(string klasa)
    {
        return new CvorStabla { 
            Klasa = klasa,
            Atribut = null,
            Tip = null,
            Threshold = null,
        };
    }

    /// <summary>
    /// Kreira čvor za grananje.
    /// </summary>
    public static CvorStabla NapraviGrananjeKategorijski(string atribut)
    {
        return new CvorStabla
        {
            Klasa = null,
            Atribut = atribut,
            Tip = TipAtributa.Kategoricki,
            Threshold = null
        };
    }

    /// <summary>
    /// Kreira čvor za grananje.
    /// </summary>
    public static CvorStabla NapraviGrananjeNumerijski(string atribut, double? threshold)
    {
        return new CvorStabla
        {
            Klasa = null,
            Atribut = atribut,
            Tip = TipAtributa.Numericki,
            Threshold = threshold
        };
    }
}



/// <summary>
/// Predstavlja implementaciju stabla odlučivanja za klasifikaciju.
/// Koristi Gini indeks i podržava samo kategorijske atribute.
/// </summary>

public class StabloKlasifikator : IKlasifikator
{
    public readonly CvorStabla korijen;

    public class StabloKlasifikatorParamteri
    {
        public int MaxDepth { get; set; } = 10;
        public int MinSamples { get; set; } = 1;
        public int BrojGrupaZaNumericke { get; set; } = 5;
    }
    public StabloKlasifikatorParamteri StabloParamteri { get; }

    public StabloKlasifikator(MojDataSet podaci, StabloKlasifikatorParamteri stabloParamteri): base(nameof(StabloKlasifikator), stabloParamteri)
    {
        ArgumentNullException.ThrowIfNull(podaci, nameof(podaci));
        ArgumentNullException.ThrowIfNull(stabloParamteri, nameof(stabloParamteri));
        var stopwatchTreniranje = System.Diagnostics.Stopwatch.StartNew();
        StabloParamteri = stabloParamteri;
        korijen = IzgradiStabloRekurzija(podaci.Podaci, podaci.Atributi.Where(x=>x.KoristiZaModel).ToList(), trenutnaDubina: 0);
        stopwatchTreniranje.Stop();
        this.VrijemeTreniranjaSek = stopwatchTreniranje.ElapsedMilliseconds / 1000.0;
        base.DodatniInfo["VaznostAtributa"] = DajVaznostAtributa();
        base.DodatniInfo["DubinaStabla"] = IzracunajDubinu(korijen);
        base.DodatniInfo["BrojCvorova"] = BrojiCvorove(korijen);
    }
    private int BrojiCvorove(CvorStabla cvor)
    {
        if (cvor.JeList)
            return 1;

        return 1 + cvor.Djeca.Values.Sum(BrojiCvorove);
    }
    private int IzracunajDubinu(CvorStabla cvor)
    {
        if (cvor.JeList)
            return 1;

        return 1 + cvor.Djeca.Values
            .Select(IzracunajDubinu)
            .DefaultIfEmpty(0)
            .Max();
    }

    private CvorStabla IzgradiStabloRekurzija(
     List<RedPodatka> podaci,
     List<AtributMeta> atributi,
     int trenutnaDubina = 0)
    {
        if (!atributi.Any())
            return CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));

        if (podaci.Count < StabloParamteri.MinSamples || trenutnaDubina >= StabloParamteri.MaxDepth)
            return CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));

        if (podaci.Select(p => p.Klasa).Distinct().Count() == 1)
            return CvorStabla.NapraviList(klasa: podaci[0].Klasa);

        string? najboljiAtribut = null;
        double najboljiGini = double.MaxValue;
        double? najboljiThreshold = null;
        bool najboljiJeNumericki = false;

        foreach (var atribut in atributi)
        {
            double gini;
            double? threshold = null;

            if (atribut.TipAtributa == TipAtributa.Kategoricki)
            {
                gini = GiniHelper.IzracunajGiniIndeks(podaci, atribut.Naziv);
            }
            else if (atribut.TipAtributa == TipAtributa.Numericki)
            {
                List<double> vrijednosti = MojDataSetHelper.DohvatiBrojeve(podaci, atribut.Naziv);
                List<double> kvantilniThresholdovi = KvartilaHelper.KvantilniThresholdovi(vrijednosti, brojGrupa: StabloParamteri.BrojGrupaZaNumericke);
                (gini, threshold) = GiniHelper.IzracunajGiniSaThresholdovima(podaci, atribut.Naziv, kvantilniThresholdovi);
            }
            else continue;

            if (gini < najboljiGini)
            {
                najboljiAtribut = atribut.Naziv;
                najboljiGini = gini;
                najboljiThreshold = threshold;
                najboljiJeNumericki = atribut.TipAtributa == TipAtributa.Numericki;
            }
        }

        if (najboljiAtribut == null)
            return CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));

        var preostali = atributi.Where(a => a.Naziv != najboljiAtribut).ToList();

        AtributMeta cvorAtribut = atributi.First(a => a.Naziv == najboljiAtribut);

        if (cvorAtribut.TipAtributa == TipAtributa.Numericki)
        {
            if (!najboljiThreshold.HasValue)
                return CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));

            CvorStabla cvor = CvorStabla.NapraviGrananjeNumerijski(najboljiAtribut, najboljiThreshold);

            var lijevi = podaci
                .Where(p => p.Atributi[najboljiAtribut].Broj.HasValue &&
                            p.Atributi[najboljiAtribut].Broj.Value <= cvor.Threshold.Value)
                .ToList();

            var desni = podaci
                .Where(p => p.Atributi[najboljiAtribut].Broj.HasValue &&
                            p.Atributi[najboljiAtribut].Broj.Value > cvor.Threshold.Value)
                .ToList();

            cvor.Djeca["<="] = lijevi.Any()
                ? IzgradiStabloRekurzija(lijevi, preostali, trenutnaDubina + 1)
                : CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));

            cvor.Djeca[">"] = desni.Any()
                ? IzgradiStabloRekurzija(desni, preostali, trenutnaDubina + 1)
                : CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));

            return cvor;
        }

        if (cvorAtribut.TipAtributa == TipAtributa.Kategoricki)
        {
            CvorStabla cvor = CvorStabla.NapraviGrananjeKategorijski(najboljiAtribut);

            var vrijednosti = podaci
                .Select(p => p.Atributi[najboljiAtribut].Tekst)
                .Where(v => !string.IsNullOrEmpty(v))
                .Distinct()
                .ToArray();

            foreach (var vr in vrijednosti)
            {
                if (vr != null)
                {
                    var podskup = podaci
                        .Where(p => p.Atributi[najboljiAtribut].Tekst == vr)
                        .ToList();

                    cvor.Djeca[vr] = podskup.Any()
                        ? IzgradiStabloRekurzija(podskup, preostali, trenutnaDubina + 1)
                        : CvorStabla.NapraviList(klasa: GiniHelper.NajcescaKlasa(podaci));
                }
            }
            return cvor;
        }

        throw new InvalidOperationException("Nepoznat tip atributa.");
    }


    public override string Predikcija(Dictionary<string, VrijednostAtributa> atributi)
    {
        return PredikcijaRekurzivno(korijen, atributi);
    }

    private string PredikcijaRekurzivno(CvorStabla cvor, Dictionary<string, VrijednostAtributa> atributi)
    {
        if (cvor.JeList)
            return cvor.Klasa!;

        if (!atributi.ContainsKey(cvor.Atribut!))
            return "Nepoznato";

        var vrijednost = atributi[cvor.Atribut!];

        if (cvor.IsNumericki && cvor.Threshold.HasValue)
        {
            if (!vrijednost.Broj.HasValue)
                return "Nepoznato";

            string grana = vrijednost.Broj.Value <= cvor.Threshold.Value ? "<=" : ">";
            if (cvor.Djeca.TryGetValue(grana, out var dijete))
                return PredikcijaRekurzivno(dijete, atributi);
        }
        else
        {
            string kljuc = vrijednost.Tekst ?? "";
            if (cvor.Djeca.TryGetValue(kljuc, out var dijete))
                return PredikcijaRekurzivno(dijete, atributi);
        }

        return "Nepoznato";
    }

    public Dictionary<string, int> DajVaznostAtributa()
    {
        var brojac = new Dictionary<string, int>();
        BrojiAtributeUStablu(korijen, brojac);
        return brojac
         .OrderByDescending(kv => kv.Value)
         .ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    private void BrojiAtributeUStablu(CvorStabla cvor, Dictionary<string, int> brojac)
    {
        if (cvor == null || cvor.JeList) return;

        if (!string.IsNullOrEmpty(cvor.Atribut))
        {
            if (!brojac.ContainsKey(cvor.Atribut))
                brojac[cvor.Atribut] = 0;
            brojac[cvor.Atribut]++;
        }

        foreach (var dijete in cvor.Djeca.Values)
        {
            BrojiAtributeUStablu(dijete, brojac);
        }
    }


}


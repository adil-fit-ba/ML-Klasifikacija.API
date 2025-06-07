using DecisionTree.Model.Model;

public class RandomForestKlasifikator : KlasifikatorBase
{
    public readonly List<StabloKlasifikator> stabla;
    private readonly Random _random;

    public class RandomForestParametri
    {
        public int BrojStabala { get; set; } = 10;
        public int? BrojAtributa { get; set; } // null znači koristi sve atribute, inače "feature bagging!"
        public StabloKlasifikator.StabloKlasifikatorParamteri ParametriStabla { get; set; } = new();
        public int? RandomState { get; set; } = 42;
    }

    public RandomForestKlasifikator(MojDataSet podaci, RandomForestParametri parametri) : base(nameof(RandomForestKlasifikator), parametri)
    {
        ArgumentNullException.ThrowIfNull(podaci);
        ArgumentNullException.ThrowIfNull(parametri);

        _random = new Random(parametri.RandomState ?? DateTime.Now.Ticks.GetHashCode());

        stabla = new List<StabloKlasifikator>();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < parametri.BrojStabala; i++)
        {
            // 1️- Bootstrap sample
            List<RedPodatka> bootstrapPodaci = new List<RedPodatka>();
            for (int j = 0; j < podaci.Podaci.Count; j++)
            {
                var randomIndex = _random.Next(podaci.Podaci.Count); // nasumičan red
                bootstrapPodaci.Add(podaci.Podaci[randomIndex]);     // dodaj u novi skup
            }

            var atributi = podaci.Atributi.Where(a => a.KoristiZaModel).ToList();
            // 2- Random atributi "feature bagging!"
            if (parametri.BrojAtributa.HasValue)
            {
                atributi = atributi.OrderBy(_ => _random.Next()).Take(parametri.BrojAtributa.Value).ToList();
            }

            AtributMeta ciljniAtribut = podaci.Atributi.First(x => x.Naziv == podaci.CiljnaKolona);

            atributi.Add(ciljniAtribut.Clone()); // dodaj ciljnu kolonu

            var podskup = new MojDataSet([..podaci.Historija, "bootstrap sampling"],bootstrapPodaci, atributi, podaci.CiljnaKolona);
            var stablo = new StabloKlasifikator(podskup, parametri.ParametriStabla);
            stabla.Add(stablo);
        }

        stopwatch.Stop();
        this.VrijemeTreniranjaSek = stopwatch.ElapsedMilliseconds / 1000.0;
        this.DodatniInfo["BrojStabala"] = stabla.Count;
    }

    public override string Predikcija(Dictionary<string, VrijednostAtributa> noviCase)
    {
        // Glasanje većine
        // Za svako stablo "s", poziva Predikcija(atributi).
        // svako stablo daje neku predikciju, tako da će rezultat ovog biti niz predikcija (niz klasa), npr
        // "Low", Low", "High", "Low", "Medium", "Low", "Medium"
        // finalni rezultat predikcije RandomForsta jeste većina glasova, tj. "Low" za prethodni slučaj.
        string[] glasovi = stabla.Select(s => s.Predikcija(noviCase)).ToArray(); 

        return glasovi 
            .GroupBy(x => x) // grupiše predikcije po klasi
            .OrderByDescending(g => g.Count()) // sortira po veličini grupe
            .First().Key; // uzima klasu koja ima najviše glasova
    }
}

using DecisionTree.Model.Model.MLP.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP;

public class MLPKlasifikator : KlasifikatorBase
{
    private readonly MLPKlasifikatorMreza mreza;
    private readonly List<AtributMeta> atributi;
    public AtributMeta CiljnaKolona { get; private set; }

    public class MLPParametri
    {
        public required int[] SkriveniSlojevi { get; set; } = [4, 4];
    }

    public MLPParametri ParametriMLP { get; }

    public MLPKlasifikator(MojDataSet podaci, MLPParametri parametri)
        : base(nameof(MLPKlasifikator), parametri)
    {
        atributi = podaci.Atributi;
        CiljnaKolona = podaci.CiljnaKolonaMeta;
        ParametriMLP = parametri;

        var stopwatchTreniranje = System.Diagnostics.Stopwatch.StartNew();

        int brojUlaza = IzracunajBrojUlaza(atributi);
        int[] skriveniSlojevi = parametri.SkriveniSlojevi;
        int brojIzlaza = podaci.Podaci.Select(p => p.Klasa).Distinct().Count();

        mreza = new MLPKlasifikatorMreza(
            brojUlaza,
            skriveniSlojevi,
            brojIzlaza
        );

        stopwatchTreniranje.Stop();
        this.VrijemeTreniranjaSek = stopwatchTreniranje.ElapsedMilliseconds / 1000.0;
    }

    private int IzracunajBrojUlaza(List<AtributMeta> atributi)
    {
        int broj = 0;
        foreach (var attr in atributi.Where(a => a.KoristiZaModel))
        {
            broj += attr.TipAtributa == TipAtributa.Numericki
                ? 1
                : attr.Kategoricki?.Top5Najcescih.Count ?? 0;
        }
        return broj;
    }

    public override string Predikcija(Dictionary<string, VrijednostAtributa> atributiReda)
    {
        var red = new RedPodatka(atributiReda);
        double[] input = MojDataSetHelperMLP.RedUInputVektor(red, atributi);
        double[] izlaz = mreza.Predikcija(input);

        // Izlaz možeš pretvoriti u labelu (npr. max vjerojatnost klasa)
        return InterpretirajIzlaz(izlaz);
    }

    private string InterpretirajIzlaz(double[] izlaz)
    {
        int indeksNajvece = Array.IndexOf(izlaz, izlaz.Max());
        var moguceKlase = CiljnaKolona
            .Kategoricki?
            .Top5Najcescih.Select(v => v.Vrijednost).ToList() ?? [];

        return moguceKlase.ElementAtOrDefault(indeksNajvece) ?? "Nepoznato";
    }
}

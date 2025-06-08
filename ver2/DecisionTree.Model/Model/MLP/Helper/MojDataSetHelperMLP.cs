using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.Helper;

/// <summary>
/// Pomoćna klasa za pretvaranje redova podataka (RedPodatka) u ulazne vektore (double[])
/// koji su spremni za korištenje u MLP mreži.
/// Ova klasa automatski radi one-hot encoding za kategorijske atribute i
/// priprema finalni niz brojeva koji neuronska mreža može obraditi.
/// </summary>
public static class MojDataSetHelperMLP
{
    /// <summary>
    /// Pretvara jedan red podataka u ulazni vektor za neuronsku mrežu.
    /// U vektor se uključuju samo atributi koji imaju oznaku "KoristiZaModel = true".
    /// Numerički atributi se dodaju direktno kao brojevi, a kategorijski se
    /// pretvaraju u one-hot kodirani niz.
    /// </summary>
    /// <param name="red">Red podataka koji se pretvara.</param>
    /// <param name="atributi">Lista meta informacija o atributima (AtributMeta).</param>
    /// <returns>Niz brojeva (double[]) koji predstavlja ulaz za neuronsku mrežu.</returns>
    public static double[] RedUInputVektor(RedPodatka red, List<AtributMeta> atributi)
    {
        var vektor = new List<double>();

        foreach (var attr in atributi.Where(a => a.KoristiZaModel))
        {
            var vrijednost = red.Atributi[attr.Naziv];

            if (attr.TipAtributa == TipAtributa.Numericki)
                vektor.Add(vrijednost.Broj ?? 0.0); // Ako je null, koristi 0
            else // kategorijski atributi – pretvaraju se u one-hot encoding
                vektor.AddRange(KategorijskiUOneHot(attr, vrijednost.Tekst));
        }

        return vektor.ToArray();
    }

    /// <summary>
    /// Pretvara vrijednost kategorijskog atributa u one-hot kodirani niz brojeva.
    /// Svaka vrijednost iz "Top5Najcescih" se poredi s trenutnom vrijednosti,
    /// i stavlja se 1.0 ako je ista, a 0.0 ako nije.
    /// </summary>
    /// <param name="attr">Meta podaci o atributu (AtributMeta).</param>
    /// <param name="vrijednost">Vrijednost atributa za trenutni red (string).</param>
    /// <returns>One-hot kodiran niz (IEnumerable&lt;double&gt;).</returns>
    private static IEnumerable<double> KategorijskiUOneHot(AtributMeta attr, string? vrijednost)
    {
        var vrijednosti = attr.Kategoricki?.Top5Najcescih.Select(v => v.Vrijednost).ToList() ?? [];
        return vrijednosti.Select(v => v == vrijednost ? 1.0 : 0.0);
    }
}

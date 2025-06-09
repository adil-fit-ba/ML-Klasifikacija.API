using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.Helper;

/// <summary>
/// Pomoćna klasa za pretvaranje redova podataka (RedPodatka) u ulazne vektore (double[])
/// koji su spremni za korištenje u MLP mreži.
/// Ova verzija pretpostavlja da su svi atributi numerički (npr. nakon one-hot kodiranja)
/// i u slučaju da naiđe na nenumerički atribut, baca Exception.
/// </summary>
public static class MojDataSetHelperMLP
{
    /// <summary>
    /// Pretvara jedan red podataka u ulazni vektor za neuronsku mrežu.
    /// U vektor se uključuju samo atributi koji imaju oznaku "KoristiZaModel = true".
    /// Očekuje se da su svi atributi numerički (npr. nakon one-hot kodiranja).
    /// </summary>
    /// <param name="atributiReda">Mapa atributa (naziv → VrijednostAtributa) za jedan red.</param>
    /// <param name="atributiMeta">Lista meta informacija o atributima koji se koriste za model.</param>
    /// <returns>Niz brojeva (double[]) koji predstavlja ulaz za neuronsku mrežu.</returns>
    /// <exception cref="InvalidOperationException">Baca se ako se naiđe na atribut koji nije numerički.</exception>
    public static double[] RedUInputVektor(Dictionary<string, VrijednostAtributa> atributiReda, AtributMeta[] atributiMeta)
    {
        double[] vektor = new double[atributiMeta.Length];

        for (int i = 0; i < atributiMeta.Length; i++)
        {
            AtributMeta atribut = atributiMeta[i];

            if (atribut.TipAtributa != TipAtributa.Numericki)
            {
                throw new InvalidOperationException($"Atribut '{atribut.Naziv}' nije numerički (pronađen: {atribut.TipAtributa}). " +
                                                    "Svi atributi moraju biti numerički – provjeri da li je izvršen one-hot encoding.");
            }

            var vrijednost = atributiReda[atribut.Naziv];
            vektor[i] = vrijednost.Broj ?? 0.0;
        }

        return vektor;
    }

    public static double[] KreirajCiljniVektor(AtributMeta atributMeta, string klasa)
    {
        var sveKlase = atributMeta.Kategoricki?.SveVrijednosti ?? [];
        var vektor = new double[sveKlase.Count];
        int indeks = sveKlase.IndexOf(klasa);
        if (indeks >= 0)
            vektor[indeks] = 1.0;
        return vektor;
    }
}

namespace DecisionTree.Model.Helper;

public static class GrupisanjeHelper
{
    /// <summary>
    ///     Grupisanje redova po dvije kategorijske kolone.
    /// </summary>
    public static Dictionary<(string, string), List<RedPodatka>> GrupisiPo2Kolone(List<RedPodatka> podaci, string kolona1, string kolona2)
    {
        return podaci
            .GroupBy(r => (
                r.GetText(kolona1) ?? string.Empty,
                r.GetText(kolona2) ?? string.Empty))
            .ToDictionary(g => g.Key, g => g.ToList());
    }

    public static Dictionary<string, List<RedPodatka>> GrupisiPo2KoloneMerged(List<RedPodatka> podaci, string kolona1, string kolona2)
    {
        return podaci
               .GroupBy(r => $"{r.GetText(kolona1) ?? string.Empty}-{r.GetText(kolona2) ?? string.Empty}")
            .ToDictionary(g => g.Key, g => g.ToList());
    }

    /// <summary>
    ///     Grupisanje redova po jednoj kategorijskoj koloni.
    /// </summary>
    public static Dictionary<string, List<RedPodatka>> GrupisiPo1Koloni(List<RedPodatka> podaci, string kolona)
    {
        return podaci
            .GroupBy(r => r.GetText(kolona) ?? string.Empty)
            .ToDictionary(g => g.Key, g => g.ToList());
    }
}

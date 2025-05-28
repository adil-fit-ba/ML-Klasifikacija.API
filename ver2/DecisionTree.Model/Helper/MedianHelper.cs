namespace DecisionTree.Model.Helper;

public static class MedianHelper
{
    /// <summary>
    ///     Izračunava medijan iz liste brojeva.
    /// </summary>
    /// <param name="vrijednosti">Lista double vrijednosti</param>
    /// <returns>Medijan ili null ako je lista prazna</returns>
    public static double? IzracunajMedijan(List<double> vrijednosti)
    {
        if (vrijednosti == null || vrijednosti.Count == 0)
            return null;

        var sortirano = vrijednosti.OrderBy(x => x).ToList();
        int n = sortirano.Count;

        if (n % 2 == 1)
            return sortirano[n / 2];
        else
            return (sortirano[n / 2 - 1] + sortirano[n / 2]) / 2.0;
    }

    /// <summary>
    ///     Izračunava medijan iz nullable double liste (null vrijednosti se ignoriraju).
    /// </summary>
    /// <param name="vrijednosti">Lista double? vrijednosti</param>
    /// <returns>Medijan ili null ako nema važećih vrijednosti</returns>
    public static double? IzracunajMedijan(List<double?> vrijednosti)
    {
        var filtrirano = vrijednosti.Where(x => x.HasValue).Select(x => x!.Value).ToList();
        return IzracunajMedijan(filtrirano);
    }
}

/// <summary>
/// Aktivacijske funkcije koje se koriste u neuronskim mrežama.
/// Klase su organizirane za skriveni sloj i za izlazni sloj mreže.
/// </summary>
public class AktivacijskeFunkcijeHelper
{
    /// <summary>
    /// Aktivacijske funkcije koje se najčešće koriste u skrivenim slojevima.
    /// Ove funkcije omogućavaju mreži da uči nelinearne odnose i da generalizira iz podataka.
    /// </summary>
    public class SkriveniSlojevi
    {
        /// <summary>
        /// ReLU (Rectified Linear Unit) funkcija: vraća maksimalno između nule i ulaza.
        /// Ako je ulaz manji od nule, izlaz je nula. Inače, izlaz je isti kao ulaz.
        /// Ovo je najčešće korištena funkcija za skrivene slojeve jer omogućava brzu i stabilnu konvergenciju.
        /// </summary>
        public static double ReLU(double x) => Math.Max(0, x);

        /// <summary>
        /// LeakyReLU funkcija: slična ReLU-u, ali za negativne ulaze vraća malu vrijednost (alpha puta x).
        /// Na ovaj način se sprječava "umiranje" neurona (dead neuron problem) koje se može javiti kod običnog ReLU-a.
        /// </summary>
        /// <param name="x">Ulazna vrijednost neurona.</param>
        /// <param name="alpha">Mali faktor za negativne vrijednosti (obično 0.01).</param>
        /// <returns>Rezultat LeakyReLU funkcije.</returns>
        public static double LeakyReLU(double x, double alpha = 0.01) =>
            x > 0 ? x : alpha * x;

        /// <summary>
        /// Tanh (hyperbolic tangent) funkcija: vraća vrijednosti između -1 i 1.
        /// Pogodna je za podatke koji su centrirani oko nule, kao kod nekih tipova sekvencijalnih mreža (npr. RNN).
        /// </summary>
        /// <param name="x">Ulazna vrijednost neurona.</param>
        /// <returns>Rezultat tanh funkcije.</returns>
        public static double Tanh(double x) =>
            Math.Tanh(x);
    }

    /// <summary>
    /// Aktivacijske funkcije za izlazni sloj mreže.
    /// Izbor funkcije zavisi od tipa problema koji mreža rješava (binarna klasifikacija, više klasa, regresija).
    /// </summary>
    public class IzlazniSlojevi
    {
        /// <summary>
        /// Sigmoid funkcija: vraća vrijednosti između 0 i 1.
        /// Koristi se za binarne klasifikacijske probleme (izlaz predstavlja vjerovatnoću jedne klase).
        /// </summary>
        /// <param name="x">Ulazna vrijednost neurona.</param>
        /// <returns>Rezultat sigmoid funkcije.</returns>
        public static double Sigmoid(double x) =>
            1.0 / (1.0 + Math.Exp(-x));

        /// <summary>
        /// Linearna aktivacijska funkcija: identitet funkcija koja vraća ulaz kao izlaz.
        /// Koristi se za regresijske probleme (npr. predikcija stvarnih brojeva kao što su cijene kuća).
        /// </summary>
        /// <param name="x">Ulazna vrijednost neurona.</param>
        /// <returns>Ista vrijednost kao ulaz.</returns>
        public static double Linear(double x) => x;

        /// <summary>
        /// Softmax funkcija: koristi se za višeklasne klasifikacijske probleme.
        /// Vraća niz vjerovatnoća koje zbrajaju 1 (svaki element predstavlja vjerovatnoću klase).
        /// </summary>
        /// <param name="x">Niz ulaznih vrijednosti za sve klase.</param>
        /// <returns>Niz vjerovatnoća za svaku klasu.</returns>
        public static double[] Softmax(double[] x)
        {
            double max = x.Max(); // Za numeričku stabilnost
            double[] exps = x.Select(xi => Math.Exp(xi - max)).ToArray();
            double sum = exps.Sum();
            return exps.Select(e => e / sum).ToArray();
        }
    }
}

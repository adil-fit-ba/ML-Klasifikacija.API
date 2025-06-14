﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.MLPMreza
{
    public class Neuron
    {
        private readonly bool Loguj = true;
        public double[] Tezine { get; set; }
        public double Bias { get; set; }
        public readonly Func<double, double> aktivacijskaFunkcija;
        public readonly Func<double, double> derivacijaAktivacijskeFunkcije;

        /// <summary>
        /// Izlazna vrijednost neurona nakon primjene aktivacijske funkcije.
        /// Koristi se za izračun delte u fazi učenja (backpropagation).
        /// Nije potrebna za predikciju (inferenciju) ako se odmah koristi rezultat.
        /// </summary>

        public double Output { get; private set; }
        /// <summary>
        ///  Delta = greška koju neuron koristi da zna koliko treba promijeniti svoje težine u sljedećem koraku treniranja, što je srž metode backpropagation.
        /// </summary>
        public double Delta { get; set; }

        public Neuron(int brojUlaza, Func<double, double> aktivacijskaFunkcija, Func<double, double> derivacija)
        {
            var rand = new Random();
            Tezine = new double[brojUlaza];
            for (int i = 0; i < brojUlaza; i++)
                Tezine[i] = rand.NextDouble() - 0.5;

            Bias = rand.NextDouble() - 0.5;
            this.aktivacijskaFunkcija = aktivacijskaFunkcija;
            this.derivacijaAktivacijskeFunkcije = derivacija;
        }

        public double Izracunaj(double[] ulazi)
        {
            double suma = 0.0;
            for (int i = 0; i < ulazi.Length; i++)
                suma += ulazi[i] * Tezine[i];

            suma += Bias;
            Output = aktivacijskaFunkcija(suma);
            return Output;
        }

        public void IzracunajDelta(double ciljnaVrijednost)
        {
            double greska = ciljnaVrijednost - Output;
            Delta = greska * derivacijaAktivacijskeFunkcije(Output);
        }

        public void IzracunajDelta(double[] tezineSljedecihNeurona, double[] deltaSljedecihNeurona)
        {
            double suma = 0.0;
            for (int i = 0; i < tezineSljedecihNeurona.Length; i++)
                suma += tezineSljedecihNeurona[i] * deltaSljedecihNeurona[i];

            Delta = suma * derivacijaAktivacijskeFunkcije(Output);
        }

        public void AzurirajTezine(double[] ulazi, double ucenjeRate)
        {
            for (int i = 0; i < Tezine.Length; i++)
                Tezine[i] += ucenjeRate * Delta * ulazi[i];

            Bias += ucenjeRate * Delta;
        }
    }
}

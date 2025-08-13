import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import Button, VBox, HBox, Label, FileUpload, Output
import io
from PIL import Image
import base64

class PotCleanlinessAnalyzer:
    def __init__(self):
        self.clean_pots = {}
        self.dirty_pots = {}
        self.analysis_results = {}
        self.current_clean_pot = None
        self.widgets_container = None

    def create_upload_interface(self):
        """Crear interfaz para subir imágenes de ollas limpias"""
        print("=== ANALIZADOR DE SUCIEDAD EN OLLAS METÁLICAS ===\n")

        # Widget para subir ollas limpias
        self.clean_upload = FileUpload(
            accept='image/*',
            multiple=True,
            description='Subir Ollas Limpias'
        )

        # Botón para confirmar ollas limpias
        self.finish_clean_btn = Button(
            description='TODAS LAS OLLAS LIMPIAS',
            button_style='success',
            layout=widgets.Layout(width='300px', height='50px')
        )

        # Output para mostrar información
        self.output = Output()

        # Conectar eventos
        self.clean_upload.observe(self.on_clean_upload, names='value')
        self.finish_clean_btn.on_click(self.finish_clean_upload)

        # Crear contenedor y mostrar
        self.widgets_container = VBox([
            Label('1. Sube las imágenes de las ollas LIMPIAS:'),
            self.clean_upload,
            self.finish_clean_btn,
            self.output
        ])

        display(self.widgets_container)

    def on_clean_upload(self, change):
        """Procesar imágenes de ollas limpias"""
        with self.output:
            clear_output()
            if change['new']:
                for filename, file_info in change['new'].items():
                    try:
                        # Cargar imagen
                        image = Image.open(io.BytesIO(file_info['content']))
                        image_array = np.array(image)

                        # Guardar imagen limpia
                        self.clean_pots[filename] = {
                            'image': image_array,
                            'filename': filename
                        }

                        print(f"✓ Olla limpia cargada: {filename}")
                    except Exception as e:
                        print(f"❌ Error cargando {filename}: {str(e)}")

                print(f"\nTotal ollas limpias: {len(self.clean_pots)}")
                if len(self.clean_pots) > 0:
                    print("Presiona 'TODAS LAS OLLAS LIMPIAS' para continuar")

    def finish_clean_upload(self, btn):
        """Terminar carga de ollas limpias e iniciar carga de sucias"""
        if len(self.clean_pots) == 0:
            with self.output:
                clear_output()
                print("❌ Error: No has subido ninguna olla limpia")
            return

        # Limpiar interfaz actual
        self.widgets_container.close()
        clear_output(wait=True)

        # Crear nueva interfaz para ollas sucias
        self.create_dirty_upload_interface()

    def create_dirty_upload_interface(self):
        """Crear interfaz para subir ollas sucias para cada olla limpia"""
        self.clean_pot_names = list(self.clean_pots.keys())
        self.current_clean_index = 0
        self.current_clean_pot = self.clean_pot_names[0]

        print(f"=== CARGANDO OLLAS SUCIAS ===")
        print(f"Total ollas limpias procesadas: {len(self.clean_pots)}\n")

        # Widget para subir ollas sucias
        self.dirty_upload = FileUpload(
            accept='image/*',
            multiple=True,
            description='Subir Ollas Sucias'
        )

        # Label dinámico
        self.current_label = Label(f'2. Sube las ollas SUCIAS para: {self.current_clean_pot}')

        # Botón para siguiente olla limpia
        self.next_clean_btn = Button(
            description='SIGUIENTE OLLA LIMPIA',
            button_style='primary',
            layout=widgets.Layout(width='250px', height='40px')
        )

        # Botón para análisis final
        self.analyze_btn = Button(
            description='INICIAR ANÁLISIS',
            button_style='warning',
            layout=widgets.Layout(width='250px', height='40px'),
            disabled=True
        )

        # Output para información
        self.dirty_output = Output()

        # Conectar eventos
        self.dirty_upload.observe(self.on_dirty_upload, names='value')
        self.next_clean_btn.on_click(self.next_clean_pot)
        self.analyze_btn.on_click(self.start_analysis)

        # Crear contenedor y mostrar
        self.widgets_container = VBox([
            self.current_label,
            self.dirty_upload,
            HBox([self.next_clean_btn, self.analyze_btn]),
            self.dirty_output
        ])

        display(self.widgets_container)

        # Inicializar diccionario para ollas sucias
        if self.current_clean_pot not in self.dirty_pots:
            self.dirty_pots[self.current_clean_pot] = {}

    def on_dirty_upload(self, change):
        """Procesar imágenes de ollas sucias"""
        with self.dirty_output:
            clear_output()
            if change['new']:
                for filename, file_info in change['new'].items():
                    try:
                        # Cargar imagen
                        image = Image.open(io.BytesIO(file_info['content']))
                        image_array = np.array(image)

                        # Guardar imagen sucia
                        self.dirty_pots[self.current_clean_pot][filename] = {
                            'image': image_array,
                            'filename': filename
                        }

                        print(f"✓ Olla sucia cargada para {self.current_clean_pot}: {filename}")
                    except Exception as e:
                        print(f"❌ Error cargando {filename}: {str(e)}")

                total_dirty = len(self.dirty_pots[self.current_clean_pot])
                print(f"\nTotal ollas sucias para {self.current_clean_pot}: {total_dirty}")

                # Verificar si podemos habilitar análisis
                self.check_analysis_ready()

    def next_clean_pot(self, btn):
        """Pasar a la siguiente olla limpia"""
        self.current_clean_index += 1

        if self.current_clean_index < len(self.clean_pot_names):
            # Siguiente olla limpia
            self.current_clean_pot = self.clean_pot_names[self.current_clean_index]

            # Actualizar label
            self.current_label.value = f'2. Sube las ollas SUCIAS para: {self.current_clean_pot}'

            # Limpiar upload widget
            self.dirty_upload.value.clear()

            # Inicializar diccionario si no existe
            if self.current_clean_pot not in self.dirty_pots:
                self.dirty_pots[self.current_clean_pot] = {}

            with self.dirty_output:
                clear_output()
                print(f"Ahora sube las ollas sucias para: {self.current_clean_pot}")

        else:
            # Todas las ollas procesadas
            self.next_clean_btn.disabled = True

            with self.dirty_output:
                clear_output()
                print("✅ Todas las ollas procesadas!")
                self.check_analysis_ready()

    def check_analysis_ready(self):
        """Verificar si se puede iniciar el análisis"""
        ready = True
        total_dirty = 0

        for clean_name in self.clean_pots.keys():
            if clean_name not in self.dirty_pots or len(self.dirty_pots[clean_name]) == 0:
                ready = False
                break
            total_dirty += len(self.dirty_pots[clean_name])

        if ready and total_dirty > 0:
            self.analyze_btn.disabled = False
            with self.dirty_output:
                print(f"✅ Listo para análisis! Total imágenes sucias: {total_dirty}")
                print("Presiona 'INICIAR ANÁLISIS'")

    def analyze_metallic_surface(self, clean_img, dirty_img):
        """Analizar el nivel de suciedad en superficie metálica"""
        try:
            # Asegurarse de que las imágenes estén en formato RGB
            if len(clean_img.shape) == 3:
                clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
            else:
                clean_gray = clean_img

            if len(dirty_img.shape) == 3:
                dirty_gray = cv2.cvtColor(dirty_img, cv2.COLOR_RGB2GRAY)
            else:
                dirty_gray = dirty_img

            # Redimensionar imágenes para comparación
            height, width = clean_gray.shape
            dirty_resized = cv2.resize(dirty_gray, (width, height))

            # Calcular diferencia absoluta
            diff = cv2.absdiff(clean_gray, dirty_resized)

            # Detectar áreas con cambios significativos (suciedad)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Calcular porcentaje de suciedad
            dirty_pixels = np.sum(thresh > 0)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            dirtiness_percentage = (dirty_pixels / total_pixels) * 100

            # Analizar brillo promedio (superficies metálicas limpias son más brillantes)
            clean_brightness = np.mean(clean_gray)
            dirty_brightness = np.mean(dirty_resized)
            brightness_reduction = max(0, clean_brightness - dirty_brightness)

            # Score combinado (mayor = más sucio)
            dirtiness_score = dirtiness_percentage + (brightness_reduction * 0.5)

            return {
                'dirtiness_percentage': float(dirtiness_percentage),
                'brightness_reduction': float(brightness_reduction),
                'dirtiness_score': float(dirtiness_score),
                'clean_brightness': float(clean_brightness),
                'dirty_brightness': float(dirty_brightness)
            }

        except Exception as e:
            print(f"Error en análisis: {e}")
            return {
                'dirtiness_percentage': 0.0,
                'brightness_reduction': 0.0,
                'dirtiness_score': 0.0,
                'clean_brightness': 0.0,
                'dirty_brightness': 0.0
            }

    def start_analysis(self, btn):
        """Iniciar análisis completo"""
        # Deshabilitar botón
        self.analyze_btn.disabled = True
        self.analyze_btn.description = "ANALIZANDO..."

        # Limpiar interfaz
        self.widgets_container.close()
        clear_output(wait=True)

        print("🔍 INICIANDO ANÁLISIS DE SUCIEDAD...")
        print("="*50)

        self.analysis_results = {}

        for clean_name, clean_data in self.clean_pots.items():
            print(f"\n📊 Analizando {clean_name}...")
            self.analysis_results[clean_name] = []

            if clean_name in self.dirty_pots:
                for dirty_name, dirty_data in self.dirty_pots[clean_name].items():
                    print(f"  - Comparando con {dirty_name}", end="...")

                    # Realizar análisis
                    analysis = self.analyze_metallic_surface(
                        clean_data['image'],
                        dirty_data['image']
                    )

                    analysis['dirty_filename'] = dirty_name
                    analysis['clean_filename'] = clean_name
                    self.analysis_results[clean_name].append(analysis)

                    print(f" Score: {analysis['dirtiness_score']:.2f}")

                # Ordenar por nivel de suciedad (más sucio primero)
                self.analysis_results[clean_name].sort(
                    key=lambda x: x['dirtiness_score'],
                    reverse=True
                )
            else:
                print(f"  ❌ No hay ollas sucias para {clean_name}")

        print("\n✅ ANÁLISIS COMPLETADO!")
        self.show_results()

    def show_results(self):
        """Mostrar resultados del análisis"""
        print("\n" + "="*70)
        print("🏆 RESULTADOS DEL ANÁLISIS DE SUCIEDAD")
        print("="*70)

        for clean_name, results in self.analysis_results.items():
            print(f"\n🍳 OLLA LIMPIA: {clean_name}")
            print("-" * 60)
            print(f"{'Ranking':<8} {'Olla Sucia':<25} {'Score':<10} {'% Sucio':<10} {'Δ Brillo'}")
            print("-" * 60)

            for i, result in enumerate(results, 1):
                filename = result['dirty_filename'][:23] + "..." if len(result['dirty_filename']) > 25 else result['dirty_filename']
                print(f"#{i:<7} {filename:<25} {result['dirtiness_score']:<10.2f} {result['dirtiness_percentage']:<10.2f} {result['brightness_reduction']:.2f}")

        print("\n" + "="*70)

        # Crear botones para exportar
        self.create_export_buttons()

    def create_export_buttons(self):
        """Crear botones para exportar resultados"""
        print("📥 OPCIONES DE EXPORTACIÓN:")

        # Botón para CSV
        csv_btn = Button(
            description='📊 DESCARGAR CSV',
            button_style='info',
            layout=widgets.Layout(width='200px', height='50px')
        )

        # Botón para gráfico
        graph_btn = Button(
            description='📈 MOSTRAR GRÁFICO',
            button_style='success',
            layout=widgets.Layout(width='200px', height='50px')
        )

        # Conectar eventos
        csv_btn.on_click(self.export_csv)
        graph_btn.on_click(self.show_graph)

        # Mostrar botones
        export_container = HBox([csv_btn, graph_btn])
        display(export_container)

    def export_csv(self, btn):
        """Exportar resultados a CSV"""
        print("\n📊 Generando archivo CSV...")

        # Preparar datos para CSV
        csv_data = []

        for clean_name, results in self.analysis_results.items():
            for i, result in enumerate(results, 1):
                csv_data.append({
                    'Olla_Limpia': clean_name,
                    'Olla_Sucia': result['dirty_filename'],
                    'Ranking_Suciedad': i,
                    'Score_Suciedad': round(result['dirtiness_score'], 2),
                    'Porcentaje_Sucio': round(result['dirtiness_percentage'], 2),
                    'Reduccion_Brillo': round(result['brightness_reduction'], 2),
                    'Brillo_Limpio': round(result['clean_brightness'], 2),
                    'Brillo_Sucio': round(result['dirty_brightness'], 2)
                })

        # Crear DataFrame y exportar
        df = pd.DataFrame(csv_data)
        csv_filename = 'analisis_suciedad_ollas.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        print(f"✅ CSV generado exitosamente: {csv_filename}")
        print(f"Total registros: {len(csv_data)}")

        # Mostrar preview
        print("\n📋 Preview del CSV:")
        print(df.head().to_string(index=False))

        # Descargar archivo
        files.download(csv_filename)

    def show_graph(self, btn):
        """Mostrar gráfico de barras"""
        print("\n📈 Generando gráficos...")

        # Configurar matplotlib para mejor visualización
        plt.style.use('default')

        num_plots = len(self.analysis_results)
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 8 * num_plots))

        if num_plots == 1:
            axes = [axes]

        # Colores para las barras (degradado de rojo a verde)
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#689f38', '#388e3c']

        for idx, (clean_name, results) in enumerate(self.analysis_results.items()):
            ax = axes[idx]

            # Preparar datos
            dirty_names = [r['dirty_filename'][:20] + "..." if len(r['dirty_filename']) > 20
                          else r['dirty_filename'] for r in results]
            scores = [r['dirtiness_score'] for r in results]

            # Asignar colores según ranking
            bar_colors = []
            for i in range(len(scores)):
                if i < len(colors):
                    bar_colors.append(colors[i])
                else:
                    bar_colors.append(colors[-1])

            # Crear barras
            bars = ax.bar(range(len(dirty_names)), scores, color=bar_colors, alpha=0.8, edgecolor='black')

            # Configurar gráfico
            ax.set_title(f'🍳 Nivel de Suciedad - {clean_name}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Ollas Sucias', fontsize=12)
            ax.set_ylabel('Score de Suciedad', fontsize=12)
            ax.set_xticks(range(len(dirty_names)))
            ax.set_xticklabels(dirty_names, rotation=45, ha='right')

            # Agregar valores en las barras
            for i, (bar, score) in enumerate(zip(bars, scores)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                       f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

                # Agregar ranking
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'#{i+1}', ha='center', va='center',
                       color='white', fontweight='bold', fontsize=14)

            # Configurar grid y límites
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(scores) * 1.15)

            # Agregar leyenda personalizada
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, edgecolor='black')
                              for i in range(min(len(results), len(colors)))]
            legend_labels = [f'#{i+1} Ranking' for i in range(min(len(results), len(colors)))]
            ax.legend(legend_elements, legend_labels, loc='upper right')

        plt.tight_layout()
        plt.show()

        print("✅ Gráficos generados exitosamente!")

    def start(self):
        """Iniciar el programa"""
        clear_output()
        self.create_upload_interface()

# Inicializar y ejecutar el analizador
print("🚀 Iniciando Analizador de Suciedad en Ollas Metálicas...")
print("Versión 2.0 - Optimizada para Google Colab")
print("-" * 50)

analyzer = PotCleanlinessAnalyzer()
analyzer.start()

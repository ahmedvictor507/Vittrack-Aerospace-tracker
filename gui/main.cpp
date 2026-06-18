#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLabel>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <QDebug>

class DroneControl : public QWidget {
    Q_OBJECT

public:
    DroneControl(QWidget *parent = nullptr) : QWidget(parent) {
        setWindowTitle("Drone Control Center");
        setFixedSize(300, 200);

        QVBoxLayout *layout = new QVBoxLayout(this);

        QLabel *label = new QLabel("Select Tracking Module:", this);
        layout->addWidget(label);

        comboBox = new QComboBox(this);
        comboBox->addItem("VitTrack (siam_tracker.py)");
        comboBox->addItem("AerospaceTracker (edgeTam_DAM4SAM_6.py)");
        layout->addWidget(comboBox);

        launchBtn = new QPushButton("Launch Tracker", this);
        layout->addWidget(launchBtn);

        attackBtn = new QPushButton("ATTACK (OFF)", this);
        attackBtn->setCheckable(true);
        attackBtn->setStyleSheet("QPushButton:checked { background-color: red; color: white; font-weight: bold; }");
        layout->addWidget(attackBtn);

        connect(launchBtn, &QPushButton::clicked, this, &DroneControl::onLaunch);
        connect(attackBtn, &QPushButton::toggled, this, &DroneControl::onAttackToggled);

        trackerProcess = new QProcess(this);
        // Ensure attack state is reset initially
        setAttackState(false);
    }

    ~DroneControl() {
        if (trackerProcess->state() == QProcess::Running) {
            trackerProcess->terminate();
            trackerProcess->waitForFinished(1000);
        }
        setAttackState(false);
    }

private slots:
    void onLaunch() {
        if (trackerProcess->state() == QProcess::Running) {
            qDebug() << "Tracker already running.";
            return;
        }

        QString scriptName = (comboBox->currentIndex() == 0) ? "siam_tracker.py" : "edgeTam_DAM4SAM_6.py";
        QString pythonExe = "/home/orin_nano/Desktop/Projects/basic_tracking/venv/bin/python";
        QString scriptPath = "/home/orin_nano/Desktop/Projects/basic_tracking/" + scriptName;

        trackerProcess->setWorkingDirectory("/home/orin_nano/Desktop/Projects/basic_tracking");
        trackerProcess->setProcessChannelMode(QProcess::ForwardedChannels); // Forward stdout to console
        trackerProcess->start(pythonExe, QStringList() << scriptPath);
        
        qDebug() << "Launched:" << scriptPath;
    }

    void onAttackToggled(bool checked) {
        if (checked) {
            attackBtn->setText("ATTACK (ON)");
            setAttackState(true);
        } else {
            attackBtn->setText("ATTACK (OFF)");
            setAttackState(false);
        }
    }

private:
    QComboBox *comboBox;
    QPushButton *launchBtn;
    QPushButton *attackBtn;
    QProcess *trackerProcess;

    void setAttackState(bool state) {
        QFile file("/tmp/drone_attack_state");
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << (state ? "1" : "0");
            file.close();
            qDebug() << "Attack state set to:" << state;
        } else {
            qDebug() << "Failed to open IPC file.";
        }
    }
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    DroneControl window;
    window.show();
    return app.exec();
}
